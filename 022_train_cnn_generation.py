# 02_train_cnn_generation.py
"""
Train a 1D CNN denoiser on synthetic PSD data produced by:
  01_synth_make_datasets.py

This version is aligned with the PSD-denoising formulation.

Script 01 writes to:
  C:/synruns/<RUN_ID>/
    config.json
    manifest.json
    datasets/
      train_gen1_clean.npz
      train_gen2_noisey_<value>.npz   OR similar variants
      train_gen3_mix_50_50.npz
      eval_mixed_noisey.npz
    plots/

This trainer:
- Uses the most recent synthetic run by default
- Trains one model per Gen2 noise_y bucket
- Runs 3 generations per model:
    Gen1: clean-only pretraining
    Gen2: fine-tune on the selected Gen2 noise_y bucket
    Gen3: fine-tune on the mixed 50/50 dataset
- Saves stage checkpoints and final checkpoints for downstream pipeline use

Outputs:
  C:/synouts/<short_run_tag>/
    models/
      cnn_g1__noisey_<value>.pth
      cnn_g2__noisey_<value>.pth
      cnn_g3__noisey_<value>.pth
      cnn_final__noisey_<value>.pth
      final_model_index.json
    logs/
      noisey_<value>/
        g1_run_meta.json
        g2_run_meta.json
        g3_run_meta.json
        g1_loss_history.npz
        g2_loss_history.npz
        g3_loss_history.npz
    plots/
      loss_g1__noisey_<value>.png
      loss_g2__noisey_<value>.png
      loss_g3__noisey_<value>.png
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# ----------------------------
# Config
# ----------------------------

@dataclass
class TrainConfig:
    # Synthetic runs root
    runs_root: str = "C:/synruns"

    # "latest" = most recently modified run folder
    run_id: str = "latest"

    # Subdir within run folder containing NPZ datasets
    datasets_subdir: str = "datasets"

    # Output parent
    out_root: str = "C:/synouts"

    # Use GEN2_NOISE_Y_LIST from the selected run's config.json if present
    use_run_config_noise_y: bool = True

    # Fallback if not using run config
    gen2_noise_y_list: Tuple[float, ...] = (2.5e-8, 4.0e-8)

    # File naming from Script 01
    # NOTE:
    # We keep these defaults, but file discovery is now tolerant to "noisy" vs "noisey"
    # and different numeric formatting in the actual filenames.
    gen2_prefix: str = "train_gen2_noisey_"
    gen2_suffix: str = ".npz"
    gen1_filename: str = "train_gen1_clean.npz"
    gen3_filename: str = "train_gen3_mix_50_50.npz"
    eval_filename: str = "eval_mixed_noisey.npz"

    # Validation
    val_split: float = 0.15
    seed: int = 7

    # Normalization
    normalize_mode: str = "per_sample"   # "none" | "global" | "per_sample"
    eps: float = 1e-12

    # Model
    base_channels: int = 32
    kernel_size: int = 7
    dropout: float = 0.0

    # Optimization
    batch_size: int = 256
    num_workers: int = 0
    lr: float = 2e-3
    weight_decay: float = 1e-6
    max_epochs_gen1: int = 80
    max_epochs_gen2: int = 60
    max_epochs_gen3: int = 60
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 12
    min_delta: float = 1e-5

    # Device / precision
    use_amp: bool = True

    # Logging
    print_every: int = 50


# ----------------------------
# Utilities
# ----------------------------

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


def _float_to_fname_piece(x: float) -> str:
    """
    Match one common Script 01 naming style:
      2.5e-8 -> 2p5e-08
      4.0e-8 -> 4p0e-08
    """
    s = f"{x:.1e}"
    return s.replace(".", "p").replace("+", "")


def short_run_tag(run_id: str, keep: int = 32) -> str:
    """
    Short filesystem-friendly tag for Windows safety.
    """
    h = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:8]
    base = run_id[:keep]
    return f"{base}__h{h}"


def resolve_run_dir(runs_root: Path, run_id: str) -> Path:
    """
    Resolve run folder:
      - If run_id == "latest": choose most recently modified directory under runs_root
      - Else: runs_root / run_id
    """
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


def resolve_gen2_noise_y_list(cfg: TrainConfig, run_dir: Path) -> Tuple[float, ...]:
    if not cfg.use_run_config_noise_y:
        return cfg.gen2_noise_y_list

    run_cfg = load_run_config(run_dir)
    vals = run_cfg.get("GEN2_NOISE_Y_LIST", None)
    if vals is None:
        return cfg.gen2_noise_y_list
    return tuple(float(v) for v in vals)


def _canonicalize_name_for_match(name: str) -> str:
    """
    Lowercase, unify 'noisy'/'noisey', and normalize number formatting tokens
    loosely for matching.
    """
    s = name.lower()
    s = s.replace("noisy", "noisey")
    return s


def _extract_float_tokens_from_text(text: str) -> List[float]:
    """
    Extract candidate floats from text. Supports formats like:
      2.5e-08
      2.5e-8
      2p5e-08
      4p0e-08
      0.000000025
    """
    s = text.lower().replace("p", ".")
    pattern = re.compile(r"""
        [+-]?
        (?:
            (?:\d+\.\d*|\.\d+|\d+)
            (?:e[+-]?\d+)?
        )
    """, re.VERBOSE)

    vals: List[float] = []
    for m in pattern.finditer(s):
        token = m.group(0)
        try:
            vals.append(float(token))
        except ValueError:
            pass
    return vals


def _is_close_noise(a: float, b: float, rtol: float = 1e-6, atol: float = 1e-20) -> bool:
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b), 1.0))


def find_gen2_file(datasets_dir: Path, noise_y: float, prefix: str, suffix: str) -> Path:
    """
    Robustly locate the Gen2 file for a given noise_y bucket.

    It now handles:
    - 'noisy' vs 'noisey'
    - numeric tokens like 2p5e-08, 2.5e-08, 2.5e-8
    - fallback matching from any train_gen2*.npz file
    """
    if not datasets_dir.exists():
        raise FileNotFoundError(f"datasets_dir does not exist: {datasets_dir}")

    expected_piece = _float_to_fname_piece(noise_y)

    # First try exact expected path from config
    expected = datasets_dir / f"{prefix}{expected_piece}{suffix}"
    if expected.exists():
        return expected

    # Build a broad list of candidate Gen2 files
    all_npz = sorted(datasets_dir.glob("*.npz"))
    gen2_candidates = [
        p for p in all_npz
        if "train_gen2" in p.name.lower()
    ]

    # Also allow 'noisy'/'noisey' interchangeably
    exactish_matches: List[Path] = []
    for p in gen2_candidates:
        cname = _canonicalize_name_for_match(p.name)
        if expected_piece.lower() in cname:
            exactish_matches.append(p)

    if exactish_matches:
        return exactish_matches[0]

    # Try numeric parsing from filename
    numeric_matches: List[Path] = []
    for p in gen2_candidates:
        vals = _extract_float_tokens_from_text(p.stem)
        for v in vals:
            if _is_close_noise(v, noise_y):
                numeric_matches.append(p)
                break

    if len(numeric_matches) == 1:
        return numeric_matches[0]

    if len(numeric_matches) > 1:
        # Prefer one containing 'noisey'/'noisy'
        for p in numeric_matches:
            cname = _canonicalize_name_for_match(p.name)
            if "noisey" in cname:
                return p
        return numeric_matches[0]

    # Last-chance fallback:
    # if there is exactly one Gen2 file total, use it.
    if len(gen2_candidates) == 1:
        print(
            f"[WARN] Exact Gen2 noise_y match for {noise_y:.3e} not found. "
            f"Using only available Gen2 file: {gen2_candidates[0].name}"
        )
        return gen2_candidates[0]

    raise FileNotFoundError(
        f"Could not find Gen2 file for noise_y={noise_y:.3e}.\n"
        f"Tried exact path: {expected}\n"
        f"Expected numeric token: {expected_piece}\n"
        f"Available Gen2 candidates: {[p.name for p in gen2_candidates]}\n"
        f"All NPZ files in datasets/: {[p.name for p in all_npz]}"
    )


def load_pairs_from_script01_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Script 01 stores:
      Y_obs   (observed noisy PSD)
      Y_tilde (clean Gaussian PSD target)

    For denoising in PSD-space:
      X_noisy = Y_obs
      X_clean = Y_tilde
    """
    d = np.load(npz_path, allow_pickle=True)
    if "Y_obs" not in d or "Y_tilde" not in d:
        raise KeyError(f"{npz_path} missing Y_obs/Y_tilde. Found keys: {list(d.keys())}")

    X_noisy = d["Y_obs"].astype(np.float32)
    X_clean = d["Y_tilde"].astype(np.float32)

    if X_noisy.ndim != 2 or X_clean.ndim != 2:
        raise ValueError(f"{npz_path}: expected (N,F) arrays; got {X_noisy.shape}, {X_clean.shape}")
    if X_noisy.shape != X_clean.shape:
        raise ValueError(f"{npz_path}: shape mismatch {X_noisy.shape} vs {X_clean.shape}")

    return X_noisy, X_clean


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
        return (X_noisy - mu) / sig, (X_clean - mu) / sig, stats

    if mode == "per_sample":
        mu = np.mean(X_noisy, axis=1, keepdims=True)
        sig = np.std(X_noisy, axis=1, keepdims=True) + eps
        stats.update({"mu_mean": float(mu.mean()), "sigma_mean": float(sig.mean())})
        return (X_noisy - mu) / sig, (X_clean - mu) / sig, stats

    raise ValueError(f"Unknown normalize_mode='{mode}'.")


def normalize_with_existing_mode(
    X_noisy: np.ndarray,
    X_clean: np.ndarray,
    mode: str,
    stats: Dict[str, float],
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Used for external eval datasets if needed.
    """
    if mode == "none":
        return X_noisy, X_clean

    if mode == "global":
        mu = float(stats["mu"])
        sig = float(stats["sigma"]) + eps
        return (X_noisy - mu) / sig, (X_clean - mu) / sig

    if mode == "per_sample":
        mu = np.mean(X_noisy, axis=1, keepdims=True)
        sig = np.std(X_noisy, axis=1, keepdims=True) + eps
        return (X_noisy - mu) / sig, (X_clean - mu) / sig

    raise ValueError(f"Unknown normalize_mode='{mode}'.")


# ----------------------------
# Dataset
# ----------------------------

class PSDPairDataset(Dataset):
    def __init__(self, X_noisy: np.ndarray, X_clean: np.ndarray):
        self.Xn = X_noisy
        self.Xc = X_clean

    def __len__(self) -> int:
        return self.Xn.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.Xn[idx])
        y = torch.from_numpy(self.Xc[idx])
        return x.unsqueeze(0), y.unsqueeze(0)  # (1,F), (1,F)


# ----------------------------
# Model
# ----------------------------

class ResidualBlock1D(nn.Module):
    def __init__(self, ch: int, k: int, dropout: float = 0.0):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(ch, ch, kernel_size=k, padding=pad),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CNNDenoiser1D(nn.Module):
    def __init__(self, base_channels: int = 32, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        k = kernel_size
        pad = k // 2
        c = base_channels

        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock1D(c, k, dropout=dropout),
            ResidualBlock1D(c, k, dropout=dropout),
            nn.Conv1d(c, c // 2, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
            ResidualBlock1D(c // 2, k, dropout=dropout),
        )
        self.head = nn.Conv1d(c // 2, 1, kernel_size=k, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.body(h)
        return self.head(h)


# ----------------------------
# Train / Eval helpers
# ----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: Optional[DataLoader], loss_fn: nn.Module, device: torch.device) -> float:
    if loader is None:
        return float("nan")

    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
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


def stage_train(
    *,
    model: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    stage_name: str,
    stage_epochs: int,
    train_file: Path,
    eval_file: Optional[Path],
    noise_y: float,
    logs_dir: Path,
    plots_dir: Path,
    models_dir: Path,
    run_id: str,
    run_tag: str,
) -> Dict[str, Any]:
    """
    Train one stage (Gen1 / Gen2 / Gen3) and save best checkpoint for that stage.
    """
    X_noisy, X_clean = load_pairs_from_script01_npz(train_file)
    Xn_norm, Xc_norm, norm_stats = normalize_arrays(X_noisy, X_clean, cfg.normalize_mode, cfg.eps)

    train_loader, val_loader, n_tr, n_val = make_loaders_from_arrays(
        Xn_norm,
        Xc_norm,
        val_split=cfg.val_split,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )

    eval_loader: Optional[DataLoader] = None
    if eval_file is not None and eval_file.exists():
        Xe_noisy, Xe_clean = load_pairs_from_script01_npz(eval_file)
        Xe_norm, Xec_norm = normalize_with_existing_mode(
            Xe_noisy, Xe_clean, cfg.normalize_mode, norm_stats, cfg.eps
        )
        eval_ds = PSDPairDataset(Xe_norm, Xec_norm)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
        "noise_y_bucket": float(noise_y),
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
                loss = loss_fn(pred, yb)

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
                    f"[{run_tag} {stage_name} noise_y={noise_y:.3e}] "
                    f"epoch {epoch+1:03d}/{stage_epochs} "
                    f"iter {it+1:04d}/{len(train_loader)} loss {loss.item():.6e}"
                )

        train_loss = running / max(n_seen, 1)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        eval_loss = evaluate(model, eval_loader, loss_fn, device)

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        eval_hist.append(eval_loss)

        msg = (
            f"[{run_tag} {stage_name} noise_y={noise_y:.3e}] "
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
                f"[{run_tag} {stage_name} noise_y={noise_y:.3e}] early stop @ epoch {epoch+1} "
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
        plt.ylabel("MSE loss")
        plt.title(f"{stage_name} training (noise_y={noise_y:.3e})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"loss_{stage_name}__noisey_{_float_to_fname_piece(noise_y)}.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"[{run_tag} {stage_name} noise_y={noise_y:.3e}] Could not save loss plot: {e}")

    stage_ckpt = models_dir / f"cnn_{stage_name}__noisey_{_float_to_fname_piece(noise_y)}.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "run_id": run_id,
            "run_tag": run_tag,
            "noise_y_bucket": float(noise_y),
            "stage_name": stage_name,
            "best_val": float(best_val),
            "best_epoch": int(best_epoch),
            "config": asdict(cfg),
            "normalize_mode": cfg.normalize_mode,
            "normalize_stats": norm_stats,
            "train_file": str(train_file.resolve()),
            "eval_file": str(eval_file.resolve()) if (eval_file is not None and eval_file.exists()) else None,
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
    }


# ----------------------------
# Main training flow
# ----------------------------

def train_three_generations_for_noise_y(
    cfg: TrainConfig,
    run_dir: Path,
    noise_y: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    For one Gen2 noise_y bucket:
      Gen1 -> Gen2 -> Gen3
    and save a final checkpoint for downstream pipeline use.
    """
    datasets_dir = run_dir / cfg.datasets_subdir
    run_id = run_dir.name
    run_tag = short_run_tag(run_id)

    out_root = Path(cfg.out_root).resolve() / run_tag
    models_dir = out_root / "models"
    logs_dir = out_root / "logs" / f"noisey_{_float_to_fname_piece(noise_y)}"
    plots_dir = out_root / "plots"
    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)

    gen1_file = datasets_dir / cfg.gen1_filename
    gen2_file = find_gen2_file(datasets_dir, noise_y, cfg.gen2_prefix, cfg.gen2_suffix)
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

    model = CNNDenoiser1D(cfg.base_channels, cfg.kernel_size, cfg.dropout).to(device)

    stage_results = []

    stage_results.append(
        stage_train(
            model=model,
            cfg=cfg,
            device=device,
            stage_name="g1",
            stage_epochs=cfg.max_epochs_gen1,
            train_file=gen1_file,
            eval_file=eval_file if eval_file.exists() else None,
            noise_y=noise_y,
            logs_dir=logs_dir,
            plots_dir=plots_dir,
            models_dir=models_dir,
            run_id=run_id,
            run_tag=run_tag,
        )
    )

    stage_results.append(
        stage_train(
            model=model,
            cfg=cfg,
            device=device,
            stage_name="g2",
            stage_epochs=cfg.max_epochs_gen2,
            train_file=gen2_file,
            eval_file=eval_file if eval_file.exists() else None,
            noise_y=noise_y,
            logs_dir=logs_dir,
            plots_dir=plots_dir,
            models_dir=models_dir,
            run_id=run_id,
            run_tag=run_tag,
        )
    )

    stage_results.append(
        stage_train(
            model=model,
            cfg=cfg,
            device=device,
            stage_name="g3",
            stage_epochs=cfg.max_epochs_gen3,
            train_file=gen3_file,
            eval_file=eval_file if eval_file.exists() else None,
            noise_y=noise_y,
            logs_dir=logs_dir,
            plots_dir=plots_dir,
            models_dir=models_dir,
            run_id=run_id,
            run_tag=run_tag,
        )
    )

    final_ckpt = models_dir / f"cnn_final__noisey_{_float_to_fname_piece(noise_y)}.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "run_id": run_id,
            "run_tag": run_tag,
            "run_dir": str(run_dir.resolve()),
            "noise_y_bucket": float(noise_y),
            "final_stage": "g3",
            "stage_results": stage_results,
            "config": asdict(cfg),
            "normalize_mode": stage_results[-1]["normalize_mode"],
            "normalize_stats": stage_results[-1]["normalize_stats"],
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
        "noise_y_bucket": float(noise_y),
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

    gen2_noise_y_list = resolve_gen2_noise_y_list(cfg, run_dir)
    print(f"Gen2 noise_y buckets to train: {gen2_noise_y_list}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results: Dict[str, Any] = {}
    for noise_y in gen2_noise_y_list:
        print("\n" + "=" * 80)
        print(f"Training 3 generations for noise_y bucket = {noise_y:.3e} (run={run_dir.name})")
        print("=" * 80)
        res = train_three_generations_for_noise_y(cfg, run_dir, noise_y, device)
        all_results[f"{noise_y:.3e}"] = res

    run_tag = short_run_tag(run_dir.name)
    models_dir = Path(cfg.out_root).resolve() / run_tag / "models"
    ensure_dir(models_dir)

    final_index = {
        "run_id": run_dir.name,
        "run_tag": run_tag,
        "run_dir": str(run_dir.resolve()),
        "final_models": {
            k: v["final_checkpoint"] for k, v in all_results.items()
        },
    }
    with open(models_dir / "final_model_index.json", "w", encoding="utf-8") as f:
        json.dump(final_index, f, indent=2)

    print("\nDone.")
    print(f"Outputs written under: {(Path(cfg.out_root).resolve() / run_tag).resolve()}")
    print(f"Final model index: {(models_dir / 'final_model_index.json').resolve()}")


if __name__ == "__main__":
    main()