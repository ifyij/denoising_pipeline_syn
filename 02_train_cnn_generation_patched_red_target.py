
# 02_train_cnn_generation_patched_red_target.py
"""
Train a 1D CNN denoiser on synthetic PSD data produced by:
  01_synth_make_datasets_patched_red_target.py

PATCHED FOR THE "RED-CURVE" EXPERIMENT

Key changes
-----------
1) Aligned to Script 01 SNR-bucket naming:
   - train_gen2_snrXXdB.npz
   - eval_mixed_snr.npz
   - reads GEN2_SNR_DB_LIST from config.json when available

2) Uses LOG-PSD training:
      x = log10(Y_obs + eps)
      y = log10(Y_target + eps)
   so absolute floor differences remain visible.

3) Uses GLOBAL normalization by default instead of per-sample normalization.

4) Target selection priority:
      Y_target -> Y_tilde -> Y_signal
   so the lower-floor target is used when present.

5) Residual-style model in log space:
      pred = x - correction
   which biases the network toward lowering the floor rather than merely smoothing.

6) Loss emphasizes floor / tail behavior:
      total_loss = mse + floor_weighted_mse + slope_loss
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
    base_channels: int = 48
    kernel_size: int = 9
    dropout: float = 0.05
    residual_nonnegative: bool = True

    # Optimization
    batch_size: int = 192
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-6
    max_epochs_gen1: int = 80
    max_epochs_gen2: int = 70
    max_epochs_gen3: int = 70
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 14
    min_delta: float = 1e-5

    # Loss weights
    floor_loss_weight: float = 0.8
    floor_gamma: float = 1.5
    slope_loss_weight: float = 0.15

    use_amp: bool = True
    print_every: int = 50


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


def load_pairs_from_script01_npz(npz_path: Path, use_log_psd: bool, log_eps: float) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    if "Y_obs" not in d:
        raise KeyError(f"{npz_path} missing Y_obs. Found keys: {list(d.keys())}")

    X_noisy = d["Y_obs"].astype(np.float32)
    X_clean = select_target_array(d)

    if X_noisy.ndim != 2 or X_clean.ndim != 2:
        raise ValueError(f"{npz_path}: expected (N,F) arrays; got {X_noisy.shape}, {X_clean.shape}")
    if X_noisy.shape != X_clean.shape:
        raise ValueError(f"{npz_path}: shape mismatch {X_noisy.shape} vs {X_clean.shape}")

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
        return X_noisy, X_clean, stats
    if mode == "global":
        mu = float(np.mean(X_noisy))
        sig = float(np.std(X_noisy) + eps)
        stats.update({"mu": mu, "sigma": sig})
        return ((X_noisy - mu) / sig).astype(np.float32), ((X_clean - mu) / sig).astype(np.float32), stats
    if mode == "per_sample":
        mu = np.mean(X_noisy, axis=1, keepdims=True)
        sig = np.std(X_noisy, axis=1, keepdims=True) + eps
        stats.update({"mu_mean": float(mu.mean()), "sigma_mean": float(sig.mean())})
        return ((X_noisy - mu) / sig).astype(np.float32), ((X_clean - mu) / sig).astype(np.float32), stats
    raise ValueError(f"Unknown normalize_mode='{mode}'.")


def normalize_with_existing_mode(
    X_noisy: np.ndarray,
    X_clean: np.ndarray,
    mode: str,
    stats: Dict[str, float],
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return X_noisy, X_clean
    if mode == "global":
        mu = float(stats["mu"])
        sig = float(stats["sigma"]) + eps
        return ((X_noisy - mu) / sig).astype(np.float32), ((X_clean - mu) / sig).astype(np.float32)
    if mode == "per_sample":
        mu = np.mean(X_noisy, axis=1, keepdims=True)
        sig = np.std(X_noisy, axis=1, keepdims=True) + eps
        return ((X_noisy - mu) / sig).astype(np.float32), ((X_clean - mu) / sig).astype(np.float32)
    raise ValueError(f"Unknown normalize_mode='{mode}'.")


class PSDPairDataset(Dataset):
    def __init__(self, X_noisy: np.ndarray, X_clean: np.ndarray):
        self.Xn = X_noisy
        self.Xc = X_clean

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
    def __init__(self, base_channels: int = 48, kernel_size: int = 9, dropout: float = 0.0, residual_nonnegative: bool = True):
        super().__init__()
        pad = kernel_size // 2
        c = base_channels
        self.residual_nonnegative = residual_nonnegative

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.body(h)
        correction = self.head(h)
        if self.residual_nonnegative:
            correction = self.softplus(correction)
        return x - correction


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    floor_loss_weight: float,
    floor_gamma: float,
    slope_loss_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    mse = torch.mean((pred - target) ** 2)

    # Emphasize bins closer to the floor / tails.
    t_min = target.amin(dim=-1, keepdim=True)
    t_max = target.amax(dim=-1, keepdim=True)
    t_norm = (target - t_min) / (t_max - t_min + 1e-6)
    weights = 1.0 + floor_loss_weight * torch.pow(1.0 - t_norm, floor_gamma)
    floor_mse = torch.mean(weights * (pred - target) ** 2)

    if pred.shape[-1] > 1:
        dp = pred[..., 1:] - pred[..., :-1]
        dt = target[..., 1:] - target[..., :-1]
        slope = torch.mean((dp - dt) ** 2)
    else:
        slope = torch.zeros((), device=pred.device, dtype=pred.dtype)

    total = mse + floor_mse + slope_loss_weight * slope
    parts = {"mse": float(mse.detach().cpu()), "floor_mse": float(floor_mse.detach().cpu()), "slope": float(slope.detach().cpu())}
    return total, parts


@torch.no_grad()
def evaluate(model: nn.Module, loader: Optional[DataLoader], cfg: TrainConfig, device: torch.device) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        loss, _ = compute_loss(pred, yb, cfg.floor_loss_weight, cfg.floor_gamma, cfg.slope_loss_weight)
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
        ds_full, [n_tr, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader, n_tr, n_val


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

    train_loader, val_loader, n_tr, n_val = make_loaders_from_arrays(
        Xn_norm, Xc_norm, cfg.val_split, cfg.seed, cfg.batch_size, cfg.num_workers, device
    )

    eval_loader: Optional[DataLoader] = None
    if eval_file is not None and eval_file.exists():
        Xe_noisy, Xe_clean = load_pairs_from_script01_npz(eval_file, cfg.use_log_psd, cfg.log_eps)
        Xe_norm, Xec_norm = normalize_with_existing_mode(Xe_noisy, Xe_clean, cfg.normalize_mode, norm_stats, cfg.eps)
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
        "config": asdict(cfg),
        "n_train": n_tr,
        "n_val": n_val,
    }
    with open(logs_dir / f"{stage_name}_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for epoch in range(stage_epochs):
        model.train()
        running, n_seen = 0.0, 0

        for it, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                pred = model(xb)
                loss, parts = compute_loss(pred, yb, cfg.floor_loss_weight, cfg.floor_gamma, cfg.slope_loss_weight)

            scaler.scale(loss).backward()

            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            scaler.step(opt)
            scaler.update()

            bs = xb.shape[0]
            running += float(loss.item()) * bs
            n_seen += bs

            if (it + 1) % cfg.print_every == 0:
                print(
                    f"[{run_tag} {stage_name} snr={snr_db:.1f}dB] "
                    f"epoch {epoch+1:03d}/{stage_epochs} "
                    f"iter {it+1:04d}/{len(train_loader)} "
                    f"loss {loss.item():.6e} mse {parts['mse']:.3e} floor {parts['floor_mse']:.3e}"
                )

        train_loss = running / max(n_seen, 1)
        val_loss = evaluate(model, val_loader, cfg, device)
        eval_loss = evaluate(model, eval_loader, cfg, device)

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        eval_hist.append(eval_loss)

        msg = (
            f"[{run_tag} {stage_name} snr={snr_db:.1f}dB] "
            f"epoch {epoch+1:03d}: train {train_loss:.6e} | val {val_loss:.6e}"
        )
        if not math.isnan(eval_loss):
            msg += f" | eval {eval_loss:.6e}"
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
        plt.figure()
        plt.plot(train_hist, label="train")
        plt.plot(val_hist, label="val")
        if eval_loader is not None:
            plt.plot(eval_hist, label="eval")
        plt.xlabel("epoch")
        plt.ylabel("weighted log-PSD loss")
        plt.title(f"{stage_name} training (snr={snr_db:.1f} dB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"loss_{stage_name}__snr_{_snr_to_fname_piece(snr_db)}dB.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"[{run_tag} {stage_name} snr={snr_db:.1f}dB] Could not save loss plot: {e}")

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
    }


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
    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)

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
        cfg.base_channels, cfg.kernel_size, cfg.dropout, residual_nonnegative=cfg.residual_nonnegative
    ).to(device)

    stage_results = []
    stage_results.append(
        stage_train(
            model=model, cfg=cfg, device=device, stage_name="g1", stage_epochs=cfg.max_epochs_gen1,
            train_file=gen1_file, eval_file=eval_file if eval_file.exists() else None, snr_db=snr_db,
            logs_dir=logs_dir, plots_dir=plots_dir, models_dir=models_dir, run_id=run_id, run_tag=run_tag,
        )
    )
    stage_results.append(
        stage_train(
            model=model, cfg=cfg, device=device, stage_name="g2", stage_epochs=cfg.max_epochs_gen2,
            train_file=gen2_file, eval_file=eval_file if eval_file.exists() else None, snr_db=snr_db,
            logs_dir=logs_dir, plots_dir=plots_dir, models_dir=models_dir, run_id=run_id, run_tag=run_tag,
        )
    )
    stage_results.append(
        stage_train(
            model=model, cfg=cfg, device=device, stage_name="g3", stage_epochs=cfg.max_epochs_gen3,
            train_file=gen3_file, eval_file=eval_file if eval_file.exists() else None, snr_db=snr_db,
            logs_dir=logs_dir, plots_dir=plots_dir, models_dir=models_dir, run_id=run_id, run_tag=run_tag,
        )
    )

    final_ckpt = models_dir / f"cnn_final__snr_{_snr_to_fname_piece(snr_db)}dB.pth"
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
