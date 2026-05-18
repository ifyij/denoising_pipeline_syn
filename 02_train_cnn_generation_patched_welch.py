# 02_train_cnn_generation_patched_welch.py
"""
Train a 1D CNN denoiser on synthetic PSD data produced by:
  01_synth_make_datasets_patched_time_welch.py

PATCHED TO MATCH THE NEW WELCH-BASED SYNTHETIC DATASET

Major fixes
-----------
1) Compatible with new Script 01:
   - train_gen1_clean.npz
   - train_gen2_snrXXdB.npz
   - train_gen3_mix_50_50.npz
   - eval_mixed_snr.npz
   - uses Y_obs -> Y_target

2) Keeps log-PSD training, but fixes dead/flat training:
   - correction is no longer forced positive with Softplus by default
   - model can learn both downward and local upward corrections in log space

3) Loss no longer double-counts plain MSE:
   - total = weighted_mse + slope_weight * slope_loss + identity_weight * identity_loss

4) Adds gradient / correction diagnostics:
   - prints mean_abs_error, mean_abs_correction, grad_norm

5) Adds quicklook prediction plots after every training stage.

6) Keeps file naming and output placement stable:
   - C:/synouts/<short_run_tag>/models/
   - cnn_g1__snr_XXdB.pth
   - cnn_g2__snr_XXdB.pth
   - cnn_g3__snr_XXdB.pth
   - cnn_final__snr_XXdB.pth
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class TrainConfig:
    runs_root: str = "C:/synruns"
    run_id: str = "latest"
    datasets_subdir: str = "datasets"
    out_root: str = "C:/synouts"

    use_run_config_snr_db: bool = True
    gen2_snr_db_list: Tuple[float, ...] = (3.0, 0.0, -3.0, -6.0)

    gen1_filename: str = "train_gen1_clean.npz"
    gen3_filename: str = "train_gen3_mix_50_50.npz"
    eval_filename: str = "eval_mixed_snr.npz"

    val_split: float = 0.15
    seed: int = 7

    # Transform / normalization
    use_log_psd: bool = True
    log_eps: float = 1e-18
    normalize_mode: str = "global"   # "none" | "global" | "per_sample"
    eps: float = 1e-12

    # Model
    base_channels: int = 64
    kernel_size: int = 9
    dropout: float = 0.03

    # IMPORTANT:
    # Old patch used residual_nonnegative=True, forcing correction >= 0.
    # That made training brittle/dead. Keep this False.
    residual_mode: str = "residual"   # "residual" or "direct"
    residual_nonnegative: bool = False

    # If residual_mode="residual":
    #     pred = x - correction_scale * correction
    correction_scale: float = 1.0

    # Optimization
    batch_size: int = 192
    num_workers: int = 0
    lr: float = 8e-4
    weight_decay: float = 1e-6

    max_epochs_gen1: int = 60
    max_epochs_gen2: int = 70
    max_epochs_gen3: int = 70

    grad_clip_norm: float = 1.0
    early_stop_patience: int = 14
    min_delta: float = 1e-5

    # Loss weights
    floor_loss_weight: float = 0.75
    floor_gamma: float = 1.25
    slope_loss_weight: float = 0.05
    linear_area_loss_weight: float = 0.20
    linear_peak_loss_weight: float = 0.15
    noise_floor_loss_weight: float = 0.08

    # Mild penalty to avoid unnecessary movement on clean Gen1.
    # Set to 0.0 if you want more aggressive correction.
    identity_loss_weight_gen1: float = 0.10
    identity_loss_weight_gen2: float = 0.00
    identity_loss_weight_gen3: float = 0.02

    # Keep AMP available, but compute fragile PSD inverse/loss terms in fp32.
    use_amp: bool = True
    print_every: int = 25

    # Quicklook plots
    n_quicklook: int = 6
    quicklook_max_freq_hz: float = 150e3

    # Resume controls
    resume_existing: bool = True
    overwrite_existing: bool = False


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _snr_to_fname_piece(x: float) -> str:
    if x < 0:
        return f"m{abs(x):g}".replace(".", "p")
    return f"{x:g}".replace(".", "p")


def short_run_tag(run_id: str, keep: int = 32) -> str:
    h = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:8]
    return f"{run_id[:keep]}__h{h}"


def resolve_run_dir(runs_root: Path, run_id: str) -> Path:
    runs_root = runs_root.resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root does not exist: {runs_root}")

    if run_id.strip().lower() != "latest":
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
        return run_dir

    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run folders found under: {runs_root}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_gen2_snr_db_list(cfg: TrainConfig, run_dir: Path) -> Tuple[float, ...]:
    if not cfg.use_run_config_snr_db:
        return cfg.gen2_snr_db_list

    run_cfg = load_run_config(run_dir)
    vals = run_cfg.get("GEN2_SNR_DB_LIST", None)
    if vals is None:
        return cfg.gen2_snr_db_list

    return tuple(float(v) for v in vals)


def find_gen2_file(datasets_dir: Path, snr_db: float) -> Path:
    piece = _snr_to_fname_piece(snr_db)
    expected = datasets_dir / f"train_gen2_snr{piece}dB.npz"
    if expected.exists():
        return expected

    candidates = sorted(datasets_dir.glob("train_gen2_snr*dB.npz"))
    if len(candidates) == 1:
        print(f"[WARN] Exact SNR bucket file not found for {snr_db:.1f} dB. Using {candidates[0].name}")
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find Gen2 file for snr_db={snr_db:.1f}. "
        f"Expected {expected}. Available: {[p.name for p in candidates]}"
    )


def select_target_array(d: np.lib.npyio.NpzFile) -> np.ndarray:
    if "Y_target" in d:
        return d["Y_target"].astype(np.float32)
    if "Y_tilde" in d:
        return d["Y_tilde"].astype(np.float32)
    if "Y_signal" in d:
        return d["Y_signal"].astype(np.float32)
    raise KeyError(f"NPZ missing Y_target/Y_tilde/Y_signal. Found keys: {list(d.keys())}")


def load_f_axis(npz_path: Path) -> Optional[np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    if "f" in d:
        return d["f"].astype(np.float32)
    return None


def load_pairs_from_script01_npz(
    npz_path: Path,
    use_log_psd: bool,
    log_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)

    if "Y_obs" not in d:
        raise KeyError(f"{npz_path} missing Y_obs. Found keys: {list(d.keys())}")

    X_noisy = d["Y_obs"].astype(np.float32)
    X_clean = select_target_array(d)

    if X_noisy.ndim != 2 or X_clean.ndim != 2:
        raise ValueError(f"{npz_path}: expected (N,F) arrays; got {X_noisy.shape}, {X_clean.shape}")

    if X_noisy.shape != X_clean.shape:
        raise ValueError(f"{npz_path}: shape mismatch {X_noisy.shape} vs {X_clean.shape}")

    if not np.isfinite(X_noisy).all() or not np.isfinite(X_clean).all():
        raise ValueError(f"{npz_path}: found non-finite values in Y_obs/Y_target")

    X_noisy = np.maximum(X_noisy, log_eps)
    X_clean = np.maximum(X_clean, log_eps)

    if use_log_psd:
        X_noisy = np.log10(X_noisy)
        X_clean = np.log10(X_clean)

    return X_noisy.astype(np.float32), X_clean.astype(np.float32)


def normalize_arrays(
    X_noisy: np.ndarray,
    X_clean: np.ndarray,
    mode: str,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    stats: Dict[str, float] = {}

    if mode == "none":
        return X_noisy.astype(np.float32), X_clean.astype(np.float32), stats

    if mode == "global":
        # Use BOTH input and target for stable scale.
        # Old version used input only, which could badly scale lower-floor targets.
        both = np.concatenate([X_noisy.reshape(-1), X_clean.reshape(-1)])
        mu = float(np.mean(both))
        sig = float(np.std(both) + eps)
        stats.update({"mu": mu, "sigma": sig})
        return (
            ((X_noisy - mu) / sig).astype(np.float32),
            ((X_clean - mu) / sig).astype(np.float32),
            stats,
        )

    if mode == "per_sample":
        mu = np.mean(X_noisy, axis=1, keepdims=True)
        sig = np.std(X_noisy, axis=1, keepdims=True) + eps
        stats.update({"mu_mean": float(mu.mean()), "sigma_mean": float(sig.mean())})
        return (
            ((X_noisy - mu) / sig).astype(np.float32),
            ((X_clean - mu) / sig).astype(np.float32),
            stats,
        )

    raise ValueError(f"Unknown normalize_mode='{mode}'.")


def normalize_with_existing_mode(
    X_noisy: np.ndarray,
    X_clean: np.ndarray,
    mode: str,
    stats: Dict[str, float],
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return X_noisy.astype(np.float32), X_clean.astype(np.float32)

    if mode == "global":
        mu = float(stats["mu"])
        sig = float(stats["sigma"]) + eps
        return (
            ((X_noisy - mu) / sig).astype(np.float32),
            ((X_clean - mu) / sig).astype(np.float32),
        )

    if mode == "per_sample":
        mu = np.mean(X_noisy, axis=1, keepdims=True)
        sig = np.std(X_noisy, axis=1, keepdims=True) + eps
        return (
            ((X_noisy - mu) / sig).astype(np.float32),
            ((X_clean - mu) / sig).astype(np.float32),
        )

    raise ValueError(f"Unknown normalize_mode='{mode}'.")


def denormalize_array(
    x: np.ndarray,
    mode: str,
    stats: Dict[str, float],
) -> np.ndarray:
    if mode == "none":
        return x

    if mode == "global":
        return x * float(stats["sigma"]) + float(stats["mu"])

    # For quicklook only. Per-sample exact inverse is not available from global stats.
    return x


def inverse_transform_psd(
    x_transformed: np.ndarray,
    *,
    use_log_psd: bool,
    log_eps: float,
) -> np.ndarray:
    if use_log_psd:
        return np.maximum(10.0 ** x_transformed, log_eps)
    return np.maximum(x_transformed, log_eps)


class PSDPairDataset(Dataset):
    def __init__(self, X_noisy: np.ndarray, X_clean: np.ndarray):
        self.Xn = X_noisy.astype(np.float32)
        self.Xc = X_clean.astype(np.float32)

    def __len__(self) -> int:
        return self.Xn.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.Xn[idx])
        y = torch.from_numpy(self.Xc[idx])
        return x.unsqueeze(0), y.unsqueeze(0)


class ResidualBlock1D(nn.Module):
    def __init__(self, ch: int, k: int, dropout: float = 0.0):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=k, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(ch, ch, kernel_size=k, padding=pad),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CNNDenoiser1D(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        kernel_size: int = 9,
        dropout: float = 0.03,
        residual_mode: str = "residual",
        residual_nonnegative: bool = False,
        correction_scale: float = 1.0,
    ):
        super().__init__()
        if residual_mode not in {"residual", "direct"}:
            raise ValueError("residual_mode must be 'residual' or 'direct'.")

        pad = kernel_size // 2
        c = base_channels

        self.residual_mode = residual_mode
        self.residual_nonnegative = residual_nonnegative
        self.correction_scale = float(correction_scale)

        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
        )

        self.body = nn.Sequential(
            ResidualBlock1D(c, kernel_size, dropout=dropout),
            ResidualBlock1D(c, kernel_size, dropout=dropout),
            nn.Conv1d(c, c, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            ResidualBlock1D(c, kernel_size, dropout=dropout),
            nn.Conv1d(c, c // 2, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            ResidualBlock1D(c // 2, kernel_size, dropout=dropout),
        )

        self.head = nn.Conv1d(c // 2, 1, kernel_size=kernel_size, padding=pad)
        self.softplus = nn.Softplus()

        # Make the initial model close to identity for residual mode.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.body(h)
        out = self.head(h)

        if self.residual_mode == "direct":
            return out

        correction = out

        if self.residual_nonnegative:
            correction = self.softplus(correction)

        return x - self.correction_scale * correction

    def predict_correction(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.residual_mode != "residual":
                return torch.zeros_like(x)
            pred = self.forward(x)
            return x - pred


def identity_weight_for_stage(cfg: TrainConfig, stage_name: str) -> float:
    if stage_name == "g1":
        return cfg.identity_loss_weight_gen1
    if stage_name == "g2":
        return cfg.identity_loss_weight_gen2
    if stage_name == "g3":
        return cfg.identity_loss_weight_gen3
    return 0.0


def transformed_tensor_to_linear(
    x: torch.Tensor,
    *,
    use_log_psd: bool,
    log_eps: float,
    normalize_mode: str,
    normalize_stats: Optional[Dict[str, float]],
    ref_transformed: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()

    if normalize_mode == "global":
        stats = normalize_stats or {}
        x = x * float(stats["sigma"]) + float(stats["mu"])
    elif normalize_mode == "per_sample":
        ref = ref_transformed.float()
        mu = torch.mean(ref, dim=-1, keepdim=True)
        sig = torch.std(ref, dim=-1, keepdim=True, unbiased=False) + eps
        x = x * sig + mu
    elif normalize_mode not in ("none", "raw"):
        raise ValueError(f"Unknown normalize_mode='{normalize_mode}'.")

    if use_log_psd:
        # Linear-domain auxiliary losses are only meant to compare physical PSDs.
        # Keep this in fp32 and clamp to a broad PSD range; fp16 10**x can overflow
        # inside AMP and was the likely source of week-long runs turning into NaNs.
        x = torch.clamp(x, min=-30.0, max=0.0)
        y = torch.pow(torch.full_like(x, 10.0, dtype=torch.float32), x)
        return torch.clamp(y, min=log_eps).to(orig_dtype if orig_dtype == torch.float64 else torch.float32)

    return torch.clamp(x, min=log_eps).to(orig_dtype if orig_dtype == torch.float64 else torch.float32)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_in: torch.Tensor,
    floor_loss_weight: float,
    floor_gamma: float,
    slope_loss_weight: float,
    identity_loss_weight: float,
    linear_area_loss_weight: float,
    linear_peak_loss_weight: float,
    noise_floor_loss_weight: float,
    *,
    use_log_psd: bool,
    log_eps: float,
    normalize_mode: str,
    normalize_stats: Optional[Dict[str, float]],
    eps: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred = pred.float()
    target = target.float()
    x_in = x_in.float()

    if not torch.isfinite(pred).all():
        raise FloatingPointError("Model produced non-finite predictions before loss computation.")
    if not torch.isfinite(target).all() or not torch.isfinite(x_in).all():
        raise FloatingPointError("Batch contains non-finite input/target values.")

    err2 = (pred - target) ** 2

    t_min = target.amin(dim=-1, keepdim=True)
    t_max = target.amax(dim=-1, keepdim=True)
    t_norm = (target - t_min) / (t_max - t_min + 1e-6)

    weights = 1.0 + floor_loss_weight * torch.pow(1.0 - t_norm, floor_gamma)
    weighted_mse = torch.mean(weights * err2)
    plain_mse = torch.mean(err2)

    if pred.shape[-1] > 1:
        dp = pred[..., 1:] - pred[..., :-1]
        dt = target[..., 1:] - target[..., :-1]
        slope = torch.mean((dp - dt) ** 2)
    else:
        slope = torch.zeros((), device=pred.device, dtype=pred.dtype)

    identity = torch.mean((pred - x_in) ** 2)

    pred_lin = transformed_tensor_to_linear(
        pred,
        use_log_psd=use_log_psd,
        log_eps=log_eps,
        normalize_mode=normalize_mode,
        normalize_stats=normalize_stats,
        ref_transformed=x_in,
        eps=eps,
    )
    target_lin = transformed_tensor_to_linear(
        target,
        use_log_psd=use_log_psd,
        log_eps=log_eps,
        normalize_mode=normalize_mode,
        normalize_stats=normalize_stats,
        ref_transformed=x_in,
        eps=eps,
    )

    area_pred = torch.sum(pred_lin, dim=-1)
    area_target = torch.sum(target_lin, dim=-1)
    peak_pred = torch.amax(pred_lin, dim=-1)
    peak_target = torch.amax(target_lin, dim=-1)

    rel_area = (area_pred - area_target) / (area_target + log_eps)
    rel_peak = (peak_pred - peak_target) / (peak_target + log_eps)
    linear_area = torch.mean(rel_area ** 2)
    linear_peak = torch.mean(rel_peak ** 2)

    n_tail = max(8, pred_lin.shape[-1] // 5)
    floor_pred = torch.mean(pred_lin[..., -n_tail:], dim=-1)
    floor_target = torch.mean(target_lin[..., -n_tail:], dim=-1)
    floor_rel = torch.log10((floor_pred + log_eps) / (floor_target + log_eps))
    noise_floor = torch.mean(floor_rel ** 2)

    total = (
        weighted_mse
        + slope_loss_weight * slope
        + identity_loss_weight * identity
        + linear_area_loss_weight * linear_area
        + linear_peak_loss_weight * linear_peak
        + noise_floor_loss_weight * noise_floor
    )

    if not torch.isfinite(total):
        raise FloatingPointError(
            "Non-finite loss. "
            f"weighted_mse={float(weighted_mse.detach().cpu()):.6e}, "
            f"slope={float(slope.detach().cpu()):.6e}, "
            f"identity={float(identity.detach().cpu()):.6e}, "
            f"linear_area={float(linear_area.detach().cpu()):.6e}, "
            f"linear_peak={float(linear_peak.detach().cpu()):.6e}, "
            f"noise_floor={float(noise_floor.detach().cpu()):.6e}"
        )

    parts = {
        "plain_mse": float(plain_mse.detach().cpu()),
        "weighted_mse": float(weighted_mse.detach().cpu()),
        "slope": float(slope.detach().cpu()),
        "identity": float(identity.detach().cpu()),
        "linear_area": float(linear_area.detach().cpu()),
        "linear_peak": float(linear_peak.detach().cpu()),
        "noise_floor": float(noise_floor.detach().cpu()),
        "median_area_ratio": float(torch.median(area_pred / (area_target + log_eps)).detach().cpu()),
        "median_peak_ratio": float(torch.median(peak_pred / (peak_target + log_eps)).detach().cpu()),
        "mae": float(torch.mean(torch.abs(pred - target)).detach().cpu()),
        "mean_abs_move": float(torch.mean(torch.abs(pred - x_in)).detach().cpu()),
    }
    return total, parts


def grad_norm_l2(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        v = float(torch.sum(p.grad.detach() ** 2).cpu())
        total += v
    return math.sqrt(max(total, 0.0))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    cfg: TrainConfig,
    device: torch.device,
    stage_name: str,
    norm_stats: Dict[str, float],
) -> float:
    if loader is None:
        return float("nan")

    model.eval()
    total, n = 0.0, 0
    id_w = identity_weight_for_stage(cfg, stage_name)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        pred = model(xb)
        loss, _ = compute_loss(
            pred,
            yb,
            xb,
            cfg.floor_loss_weight,
            cfg.floor_gamma,
            cfg.slope_loss_weight,
            id_w,
            cfg.linear_area_loss_weight,
            cfg.linear_peak_loss_weight,
            cfg.noise_floor_loss_weight,
            use_log_psd=cfg.use_log_psd,
            log_eps=cfg.log_eps,
            normalize_mode=cfg.normalize_mode,
            normalize_stats=norm_stats,
            eps=cfg.eps,
        )

        bs = xb.shape[0]
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


def make_loaders_from_arrays(
    X_noisy: np.ndarray,
    X_clean: np.ndarray,
    val_split: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, int, int]:
    ds_full = PSDPairDataset(X_noisy, X_clean)
    n_total = len(ds_full)
    n_val = max(1, int(math.floor(val_split * n_total)))
    n_tr = n_total - n_val

    ds_train, ds_val = random_split(
        ds_full,
        [n_tr, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader, n_tr, n_val


@torch.no_grad()
def save_quicklook_predictions(
    *,
    model: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    npz_file: Path,
    out_path: Path,
    norm_stats: Dict[str, float],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Could not import matplotlib for quicklook: {e}")
        return

    d = np.load(npz_file, allow_pickle=True)
    f = d["f"].astype(np.float32) if "f" in d else np.arange(d["Y_obs"].shape[1], dtype=np.float32)

    Y_obs_linear = d["Y_obs"].astype(np.float32)
    Y_tgt_linear = select_target_array(d)
    Y_signal_linear = d["Y_signal"].astype(np.float32) if "Y_signal" in d else None

    X_raw, Y_raw = load_pairs_from_script01_npz(npz_file, cfg.use_log_psd, cfg.log_eps)
    X_norm, Y_norm = normalize_with_existing_mode(X_raw, Y_raw, cfg.normalize_mode, norm_stats, cfg.eps)

    model.eval()

    n = min(cfg.n_quicklook, X_norm.shape[0])
    idx = np.linspace(0, X_norm.shape[0] - 1, n, dtype=int)

    freq_mask = np.ones_like(f, dtype=bool)
    if cfg.quicklook_max_freq_hz and cfg.quicklook_max_freq_hz > 0:
        freq_mask = f <= cfg.quicklook_max_freq_hz

    for k, i in enumerate(idx):
        xb = torch.from_numpy(X_norm[i][None, None, :]).to(device)
        pred_norm = model(xb).detach().cpu().numpy()[0, 0]

        pred_trans = denormalize_array(pred_norm, cfg.normalize_mode, norm_stats)
        pred_linear = inverse_transform_psd(
            pred_trans,
            use_log_psd=cfg.use_log_psd,
            log_eps=cfg.log_eps,
        )

        plt.figure(figsize=(11, 6))
        plt.plot(f[freq_mask], Y_obs_linear[i][freq_mask], label="Y_obs", linewidth=1.5)
        plt.plot(f[freq_mask], Y_tgt_linear[i][freq_mask], "--", label="Y_target", linewidth=1.8)
        if Y_signal_linear is not None:
            plt.plot(f[freq_mask], Y_signal_linear[i][freq_mask], "-.", label="Y_signal truth", linewidth=1.3, alpha=0.9)
        plt.plot(f[freq_mask], pred_linear[freq_mask], label="CNN pred", linewidth=1.8)
        df_hz = float(np.median(np.diff(f[freq_mask]))) if np.count_nonzero(freq_mask) > 1 else 1.0
        area_ratio = float(np.sum(pred_linear[freq_mask]) / max(np.sum(Y_tgt_linear[i][freq_mask]), cfg.log_eps))
        peak_ratio = float(np.max(pred_linear[freq_mask]) / max(np.max(Y_tgt_linear[i][freq_mask]), cfg.log_eps))
        plt.xlabel("f (Hz)")
        plt.ylabel(r"PSD (V$^2$/Hz)")
        plt.title(f"{title} quicklook ex {i} | area={area_ratio:.3f}, peak={peak_ratio:.3f}, df={df_hz:.1f}Hz")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path.parent / f"{out_path.stem}_ex_{k}_idx_{i}.png", dpi=200)
        plt.close()


def stage_train(
    *,
    model: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    stage_name: str,
    stage_epochs: int,
    train_file: Path,
    eval_file: Optional[Path],
    snr_db: float,
    logs_dir: Path,
    plots_dir: Path,
    models_dir: Path,
    run_id: str,
    run_tag: str,
) -> Dict[str, Any]:
    X_noisy, X_clean = load_pairs_from_script01_npz(train_file, cfg.use_log_psd, cfg.log_eps)
    Xn_norm, Xc_norm, norm_stats = normalize_arrays(X_noisy, X_clean, cfg.normalize_mode, cfg.eps)

    print(
        f"[{run_tag} {stage_name} snr={snr_db:.1f}dB] "
        f"raw input mean/std={X_noisy.mean():.4e}/{X_noisy.std():.4e}, "
        f"raw target mean/std={X_clean.mean():.4e}/{X_clean.std():.4e}"
    )
    print(
        f"[{run_tag} {stage_name} snr={snr_db:.1f}dB] "
        f"norm input mean/std={Xn_norm.mean():.4e}/{Xn_norm.std():.4e}, "
        f"norm target mean/std={Xc_norm.mean():.4e}/{Xc_norm.std():.4e}"
    )

    train_loader, val_loader, n_tr, n_val = make_loaders_from_arrays(
        Xn_norm,
        Xc_norm,
        cfg.val_split,
        cfg.seed,
        cfg.batch_size,
        cfg.num_workers,
        device,
    )

    eval_loader: Optional[DataLoader] = None
    if eval_file is not None and eval_file.exists():
        Xe_noisy, Xe_clean = load_pairs_from_script01_npz(eval_file, cfg.use_log_psd, cfg.log_eps)
        Xe_norm, Xec_norm = normalize_with_existing_mode(
            Xe_noisy,
            Xe_clean,
            cfg.normalize_mode,
            norm_stats,
            cfg.eps,
        )
        eval_loader = DataLoader(
            PSDPairDataset(Xe_norm, Xec_norm),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0

    train_hist: List[float] = []
    val_hist: List[float] = []
    eval_hist: List[float] = []

    id_w = identity_weight_for_stage(cfg, stage_name)

    meta = {
        "run_id": run_id,
        "run_tag": run_tag,
        "stage_name": stage_name,
        "train_file": str(train_file.resolve()),
        "eval_file": str(eval_file.resolve()) if (eval_file is not None and eval_file.exists()) else None,
        "snr_db_bucket": float(snr_db),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "normalize_mode": cfg.normalize_mode,
        "normalize_stats": norm_stats,
        "use_log_psd": bool(cfg.use_log_psd),
        "log_eps": float(cfg.log_eps),
        "config": asdict(cfg),
        "n_train": n_tr,
        "n_val": n_val,
    }

    with open(logs_dir / f"{stage_name}_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for epoch in range(stage_epochs):
        model.train()
        running, n_seen = 0.0, 0

        last_parts: Dict[str, float] = {}
        last_grad_norm = 0.0

        for it, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                pred = model(xb)
                loss, parts = compute_loss(
                    pred,
                    yb,
                    xb,
                    cfg.floor_loss_weight,
                    cfg.floor_gamma,
                    cfg.slope_loss_weight,
                    id_w,
                    cfg.linear_area_loss_weight,
                    cfg.linear_peak_loss_weight,
                    cfg.noise_floor_loss_weight,
                    use_log_psd=cfg.use_log_psd,
                    log_eps=cfg.log_eps,
                    normalize_mode=cfg.normalize_mode,
                    normalize_stats=norm_stats,
                    eps=cfg.eps,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(opt)
            last_grad_norm = grad_norm_l2(model)
            if not math.isfinite(last_grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at {stage_name} epoch {epoch+1}, iter {it+1}."
                )

            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.grad_clip_norm,
                    error_if_nonfinite=True,
                )

            scaler.step(opt)
            scaler.update()

            bs = xb.shape[0]
            running += float(loss.item()) * bs
            n_seen += bs
            last_parts = parts

            if (it + 1) % cfg.print_every == 0:
                print(
                    f"[{run_tag} {stage_name} snr={snr_db:.1f}dB] "
                    f"epoch {epoch+1:03d}/{stage_epochs} "
                    f"iter {it+1:04d}/{len(train_loader)} "
                    f"loss {loss.item():.6e} "
                    f"wmse {parts['weighted_mse']:.3e} "
                    f"mse {parts['plain_mse']:.3e} "
                    f"mae {parts['mae']:.3e} "
                    f"area {parts['median_area_ratio']:.3f} "
                    f"peak {parts['median_peak_ratio']:.3f} "
                    f"move {parts['mean_abs_move']:.3e} "
                    f"grad {last_grad_norm:.3e}"
                )

        train_loss = running / max(n_seen, 1)
        val_loss = evaluate(model, val_loader, cfg, device, stage_name, norm_stats)
        eval_loss = evaluate(model, eval_loader, cfg, device, stage_name, norm_stats)

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        eval_hist.append(eval_loss)

        msg = (
            f"[{run_tag} {stage_name} snr={snr_db:.1f}dB] "
            f"epoch {epoch+1:03d}: train {train_loss:.6e} | val {val_loss:.6e}"
        )
        if not math.isnan(eval_loss):
            msg += f" | eval {eval_loss:.6e}"
        if last_parts:
            msg += (
                f" | mae {last_parts.get('mae', float('nan')):.3e}"
                f" | area {last_parts.get('median_area_ratio', float('nan')):.3f}"
                f" | peak {last_parts.get('median_peak_ratio', float('nan')):.3f}"
                f" | move {last_parts.get('mean_abs_move', float('nan')):.3e}"
                f" | grad {last_grad_norm:.3e}"
            )
        print(msg)

        improved = (best_val - val_loss) > cfg.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if patience >= cfg.early_stop_patience:
            print(
                f"[{run_tag} {stage_name} snr={snr_db:.1f}dB] early stop @ epoch {epoch+1} "
                f"(best epoch {best_epoch+1}, best val {best_val:.6e})"
            )
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    np.savez(
        logs_dir / f"{stage_name}_loss_history.npz",
        train=np.asarray(train_hist, dtype=np.float32),
        val=np.asarray(val_hist, dtype=np.float32),
        eval=np.asarray(eval_hist, dtype=np.float32),
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(train_hist, label="train")
        plt.plot(val_hist, label="val")
        if eval_loader is not None:
            plt.plot(eval_hist, label="eval")
        plt.xlabel("epoch")
        plt.ylabel("weighted loss")
        plt.title(f"{stage_name} training (snr={snr_db:.1f} dB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"loss_{stage_name}__snr_{_snr_to_fname_piece(snr_db)}dB.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"[{run_tag} {stage_name} snr={snr_db:.1f}dB] Could not save loss plot: {e}")

    quicklook_base = plots_dir / f"quicklook_{stage_name}__snr_{_snr_to_fname_piece(snr_db)}dB.png"
    save_quicklook_predictions(
        model=model,
        cfg=cfg,
        device=device,
        npz_file=train_file,
        out_path=quicklook_base,
        norm_stats=norm_stats,
        title=f"{stage_name} train snr={snr_db:.1f}dB",
    )

    if eval_file is not None and eval_file.exists():
        quicklook_eval = plots_dir / f"quicklook_{stage_name}__snr_{_snr_to_fname_piece(snr_db)}dB_eval.png"
        save_quicklook_predictions(
            model=model,
            cfg=cfg,
            device=device,
            npz_file=eval_file,
            out_path=quicklook_eval,
            norm_stats=norm_stats,
            title=f"{stage_name} eval snr={snr_db:.1f}dB",
        )

    stage_ckpt = models_dir / f"cnn_{stage_name}__snr_{_snr_to_fname_piece(snr_db)}dB.pth"

    torch.save(
        {
            "model_state": model.state_dict(),
            "run_id": run_id,
            "run_tag": run_tag,
            "snr_db_bucket": float(snr_db),
            "stage_name": stage_name,
            "best_val": float(best_val),
            "best_epoch": int(best_epoch),
            "config": asdict(cfg),
            "normalize_mode": cfg.normalize_mode,
            "normalize_stats": norm_stats,
            "train_file": str(train_file.resolve()),
            "eval_file": str(eval_file.resolve()) if (eval_file is not None and eval_file.exists()) else None,
            "use_log_psd": bool(cfg.use_log_psd),
            "log_eps": float(cfg.log_eps),
            "residual_mode": cfg.residual_mode,
            "residual_nonnegative": bool(cfg.residual_nonnegative),
            "correction_scale": float(cfg.correction_scale),
        },
        stage_ckpt,
    )

    return {
        "stage_name": stage_name,
        "checkpoint": str(stage_ckpt),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "normalize_mode": cfg.normalize_mode,
        "normalize_stats": norm_stats,
        "use_log_psd": bool(cfg.use_log_psd),
        "log_eps": float(cfg.log_eps),
        "residual_mode": cfg.residual_mode,
        "residual_nonnegative": bool(cfg.residual_nonnegative),
        "correction_scale": float(cfg.correction_scale),
    }


def load_existing_stage_checkpoint(
    *,
    model: nn.Module,
    ckpt_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    return {
        "stage_name": ckpt.get("stage_name", ckpt_path.stem),
        "checkpoint": str(ckpt_path),
        "best_val": float(ckpt.get("best_val", float("nan"))),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "normalize_mode": ckpt.get("normalize_mode", "unknown"),
        "normalize_stats": ckpt.get("normalize_stats", {}),
        "use_log_psd": bool(ckpt.get("use_log_psd", False)),
        "log_eps": float(ckpt.get("log_eps", 1e-18)),
        "residual_mode": ckpt.get("residual_mode", "residual"),
        "residual_nonnegative": bool(ckpt.get("residual_nonnegative", False)),
        "correction_scale": float(ckpt.get("correction_scale", 1.0)),
    }


def checkpoint_is_healthy(ckpt_path: Path, device: torch.device) -> bool:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"[WARN] Could not inspect checkpoint {ckpt_path}: {exc}")
        return False

    if "best_val" in ckpt:
        best_val = float(ckpt.get("best_val", float("nan")))
        if not math.isfinite(best_val):
            print(f"[WARN] Checkpoint has non-finite best_val; will retrain: {ckpt_path}")
            return False
    else:
        for stage in ckpt.get("stage_results", []):
            best_val = float(stage.get("best_val", float("nan")))
            if not math.isfinite(best_val):
                print(f"[WARN] Checkpoint has non-finite stage best_val; will retrain: {ckpt_path}")
                return False

    state = ckpt.get("model_state")
    if not isinstance(state, dict):
        print(f"[WARN] Checkpoint missing model_state; will retrain: {ckpt_path}")
        return False

    for name, tensor in state.items():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            print(f"[WARN] Checkpoint tensor '{name}' is non-finite; will retrain: {ckpt_path}")
            return False

    return True


def train_three_generations_for_snr(
    cfg: TrainConfig,
    run_dir: Path,
    snr_db: float,
    device: torch.device,
) -> Dict[str, Any]:
    datasets_dir = run_dir / cfg.datasets_subdir
    run_id = run_dir.name
    run_tag = short_run_tag(run_id)

    out_root = Path(cfg.out_root).resolve() / run_tag
    models_dir = out_root / "models"
    logs_dir = out_root / "logs" / f"snr_{_snr_to_fname_piece(snr_db)}dB"
    plots_dir = out_root / "plots"
    final_ckpt = models_dir / f"cnn_final__snr_{_snr_to_fname_piece(snr_db)}dB.pth"

    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)

    if (
        cfg.resume_existing
        and not cfg.overwrite_existing
        and final_ckpt.exists()
        and checkpoint_is_healthy(final_ckpt, device)
    ):
        print(f"[resume] Final checkpoint already exists for snr={snr_db:.1f} dB; skipping: {final_ckpt}")
        summary_path = logs_dir / "final_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "snr_db_bucket": float(snr_db),
            "final_checkpoint": str(final_ckpt),
            "stage_results": [],
            "resumed_from_existing_final": True,
        }

    gen1_file = datasets_dir / cfg.gen1_filename
    gen2_file = find_gen2_file(datasets_dir, snr_db)
    gen3_file = datasets_dir / cfg.gen3_filename
    eval_file = datasets_dir / cfg.eval_filename

    print(f"[INFO] Using Gen1 file: {gen1_file.name}")
    print(f"[INFO] Using Gen2 file: {gen2_file.name}")
    print(f"[INFO] Using Gen3 file: {gen3_file.name}")

    if eval_file.exists():
        print(f"[INFO] Using Eval file: {eval_file.name}")
    else:
        print(f"[WARN] Eval file not found: {eval_file}")

    for req in [gen1_file, gen2_file, gen3_file]:
        if not req.exists():
            raise FileNotFoundError(f"Missing required dataset: {req}")

    model = CNNDenoiser1D(
        base_channels=cfg.base_channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        residual_mode=cfg.residual_mode,
        residual_nonnegative=cfg.residual_nonnegative,
        correction_scale=cfg.correction_scale,
    ).to(device)

    stage_results = []

    stage_specs = [
        ("g1", cfg.max_epochs_gen1, gen1_file),
        ("g2", cfg.max_epochs_gen2, gen2_file),
        ("g3", cfg.max_epochs_gen3, gen3_file),
    ]

    for stage_name, stage_epochs, train_file in stage_specs:
        stage_ckpt = models_dir / f"cnn_{stage_name}__snr_{_snr_to_fname_piece(snr_db)}dB.pth"
        if (
            cfg.resume_existing
            and not cfg.overwrite_existing
            and stage_ckpt.exists()
            and checkpoint_is_healthy(stage_ckpt, device)
        ):
            print(f"[resume] Loading existing {stage_name} checkpoint: {stage_ckpt}")
            stage_results.append(
                load_existing_stage_checkpoint(
                    model=model,
                    ckpt_path=stage_ckpt,
                    device=device,
                )
            )
            continue

        stage_results.append(
            stage_train(
                model=model,
                cfg=cfg,
                device=device,
                stage_name=stage_name,
                stage_epochs=stage_epochs,
                train_file=train_file,
                eval_file=eval_file if eval_file.exists() else None,
                snr_db=snr_db,
                logs_dir=logs_dir,
                plots_dir=plots_dir,
                models_dir=models_dir,
                run_id=run_id,
                run_tag=run_tag,
            )
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "run_id": run_id,
            "run_tag": run_tag,
            "run_dir": str(run_dir.resolve()),
            "snr_db_bucket": float(snr_db),
            "final_stage": "g3",
            "stage_results": stage_results,
            "config": asdict(cfg),
            "normalize_mode": stage_results[-1]["normalize_mode"],
            "normalize_stats": stage_results[-1]["normalize_stats"],
            "use_log_psd": stage_results[-1]["use_log_psd"],
            "log_eps": stage_results[-1]["log_eps"],
            "residual_mode": stage_results[-1]["residual_mode"],
            "residual_nonnegative": stage_results[-1]["residual_nonnegative"],
            "correction_scale": stage_results[-1]["correction_scale"],
            "train_files": {
                "g1": str(gen1_file.resolve()),
                "g2": str(gen2_file.resolve()),
                "g3": str(gen3_file.resolve()),
            },
            "eval_file": str(eval_file.resolve()) if eval_file.exists() else None,
        },
        final_ckpt,
    )

    result = {
        "snr_db_bucket": float(snr_db),
        "final_checkpoint": str(final_ckpt),
        "stage_results": stage_results,
    }

    with open(logs_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    runs_root = Path(cfg.runs_root)
    run_dir = resolve_run_dir(runs_root, cfg.run_id)

    print(f"Using synthetic run folder: {run_dir.resolve()}")

    run_cfg = load_run_config(run_dir)
    if run_cfg:
        print("[INFO] Found Script 01 config.json")
        print(f"[INFO] Script 01 generation mode may be: {run_cfg.get('generation_mode', 'not specified')}")
        print(f"[INFO] Script 01 fs={run_cfg.get('fs', 'unknown')}, nperseg={run_cfg.get('nperseg', 'unknown')}")

    gen2_snr_db_list = resolve_gen2_snr_db_list(cfg, run_dir)
    print(f"Gen2 SNR buckets to train: {gen2_snr_db_list}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results: Dict[str, Any] = {}

    for snr_db in gen2_snr_db_list:
        print("\n" + "=" * 80)
        print(f"Training 3 generations for SNR bucket = {snr_db:.1f} dB (run={run_dir.name})")
        print("=" * 80)

        res = train_three_generations_for_snr(cfg, run_dir, snr_db, device)
        all_results[f"{snr_db:.1f}dB"] = res

    run_tag = short_run_tag(run_dir.name)
    models_dir = Path(cfg.out_root).resolve() / run_tag / "models"
    ensure_dir(models_dir)

    final_index = {
        "run_id": run_dir.name,
        "run_tag": run_tag,
        "run_dir": str(run_dir.resolve()),
        "final_models": {k: v["final_checkpoint"] for k, v in all_results.items()},
    }

    with open(models_dir / "final_model_index.json", "w", encoding="utf-8") as f:
        json.dump(final_index, f, indent=2)

    print("\nDone.")
    print(f"Outputs written under: {(Path(cfg.out_root).resolve() / run_tag).resolve()}")
    print(f"Final model index: {(models_dir / 'final_model_index.json').resolve()}")


if __name__ == "__main__":
    main()
