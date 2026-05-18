# 02_train_cnn_generation.py
"""
Train a 1D CNN denoiser on synthetic PSD data produced by:
  01_synth_make_datasets.py

Script 01 outputs:
  synthetic_datasets/
    config.json
    datasets/
      train_gen1_clean.npz
      train_gen2_snr20dB.npz   (or train_gen2_snr20p0dB.npz depending on formatting)
      train_gen2_snr10dB.npz
      train_gen3_mix_50_50.npz
      eval_mixed_snr.npz
    plots/
      ...

This trainer:
- Builds training pairs from NPZ:
    input  = Y_obs   (observed PSD from Welch)
    target = Y_tilde (clean Gaussian PSD target)
- Trains ONE model per Gen2 SNR in GEN2_SNR_DB_LIST (recommended for your "crank SNR" workflow)
- Saves:
    outputs/
      models/cnn_denoiser_snr_XXdB.pth
      logs/snr_XXdB/...
      plots/loss_snr_XXdB.png
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ----------------------------
# Config
# ----------------------------

@dataclass
class TrainConfig:
    # Where script 01 wrote data
    synth_root: str = "synthetic_datasets"
    datasets_subdir: str = "datasets"

    # Output root for training artifacts
    out_root: str = "outputs"

    # Which Gen2 SNRs to train per-model on
    # If you keep these in sync with 01's GEN2_SNR_DB_LIST, you can just edit here once.
    gen2_snr_db_list: Tuple[float, ...] = (20.0, 10.0)

    # Training data file naming (script 01 convention)
    # script 01 used: fname = f"train_gen2_snr{snr:g}dB.npz".replace(".", "p")
    # So 10.0 -> "train_gen2_snr10dB.npz" (no decimal), 7.5 -> "train_gen2_snr7p5dB.npz"
    gen2_prefix: str = "train_gen2_snr"
    gen2_suffix: str = "dB.npz"

    # If you want to also include additional datasets in training, you can add them here:
    extra_train_files: Tuple[str, ...] = ()  # e.g., ("train_gen3_mix_50_50.npz",)

    # Validation
    val_split: float = 0.15
    seed: int = 7

    # Normalization
    normalize_mode: str = "per_sample"  # "none" | "global" | "per_sample"
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
    max_epochs: int = 80
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


def _snr_to_fname_piece(snr_db: float) -> str:
    """
    Match script 01 naming:
      f"{snr:g}" then replace "." with "p"
    """
    s = f"{snr_db:g}"
    return s.replace(".", "p")


def find_gen2_file(datasets_dir: Path, snr_db: float, prefix: str, suffix: str) -> Path:
    """
    Robustly locate the Gen2 file for a given SNR:
    - first try exact expected name
    - then fallback: glob any file that starts with prefix and contains the snr piece
    """
    piece = _snr_to_fname_piece(snr_db)
    expected = datasets_dir / f"{prefix}{piece}{suffix}"
    if expected.exists():
        return expected

    # fallback: try to find something close
    candidates = sorted(datasets_dir.glob(f"{prefix}*{suffix}"))
    for c in candidates:
        if piece in c.name:
            return c

    raise FileNotFoundError(
        f"Could not find Gen2 file for SNR={snr_db} dB.\n"
        f"Tried: {expected}\n"
        f"Available: {[p.name for p in candidates]}"
    )


def load_pairs_from_script01_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Script 01 stores:
      Y_obs   (observed Welch PSD)
      Y_tilde (clean Gaussian PSD target)

    For denoising in PSD-space:
      X_noisy = Y_obs
      X_clean = Y_tilde
    """
    d = np.load(npz_path, allow_pickle=True)
    if "Y_obs" not in d or "Y_tilde" not in d:
        raise KeyError(f"{npz_path} missing Y_obs/Y_tilde. Found keys: {list(d.keys())}")

    X_noisy = d["Y_obs"].astype(np.float32)    # (N, F)
    X_clean = d["Y_tilde"].astype(np.float32)  # (N, F)

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
        x = torch.from_numpy(self.Xn[idx])  # (F,)
        y = torch.from_numpy(self.Xc[idx])  # (F,)
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
# Train / Eval
# ----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        bs = xb.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


def train_for_snr(cfg: TrainConfig, snr_db: float, device: torch.device) -> None:
    synth_root = Path(cfg.synth_root).resolve()
    datasets_dir = synth_root / cfg.datasets_subdir

    # Collect files: main gen2 file + optional extras
    gen2_file = find_gen2_file(datasets_dir, snr_db, cfg.gen2_prefix, cfg.gen2_suffix)
    train_files = [gen2_file] + [datasets_dir / f for f in cfg.extra_train_files]

    Xn_list, Xc_list = [], []
    for f in train_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing training npz: {f}")
        Xn, Xc = load_pairs_from_script01_npz(f)
        Xn_list.append(Xn)
        Xc_list.append(Xc)

    X_noisy = np.concatenate(Xn_list, axis=0)
    X_clean = np.concatenate(Xc_list, axis=0)

    # Normalize
    Xn_norm, Xc_norm, norm_stats = normalize_arrays(X_noisy, X_clean, cfg.normalize_mode, cfg.eps)

    ds_full = PSDPairDataset(Xn_norm, Xc_norm)
    n_total = len(ds_full)
    n_val = max(1, int(math.floor(cfg.val_split * n_total)))
    n_tr = n_total - n_val

    ds_train, ds_val = random_split(
        ds_full,
        [n_tr, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Output dirs
    out_root = Path(cfg.out_root).resolve()
    models_dir = out_root / "models"
    logs_dir = out_root / "logs" / f"snr_{_snr_to_fname_piece(snr_db)}dB"
    plots_dir = out_root / "plots"
    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)

    model = CNNDenoiser1D(cfg.base_channels, cfg.kernel_size, cfg.dropout).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val = float("inf")
    best_epoch = -1
    patience = 0

    train_hist: List[float] = []
    val_hist: List[float] = []

    # Save run metadata
    meta = {
        "snr_db": float(snr_db),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "train_files": [str(p) for p in train_files],
        "normalize_mode": cfg.normalize_mode,
        "normalize_stats": norm_stats,
        "config": asdict(cfg),
        "n_train": n_tr,
        "n_val": n_val,
    }
    with open(logs_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    global_step = 0
    for epoch in range(cfg.max_epochs):
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
            global_step += 1

            if (it + 1) % cfg.print_every == 0:
                print(f"[snr={snr_db:g} dB] epoch {epoch+1:03d}/{cfg.max_epochs} "
                      f"iter {it+1:04d}/{len(train_loader)} loss {loss.item():.6e}")

        train_loss = running / max(n_seen, 1)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        print(f"[snr={snr_db:g} dB] epoch {epoch+1:03d}: train {train_loss:.6e} | val {val_loss:.6e}")

        improved = (best_val - val_loss) > cfg.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience = 0

            ckpt = models_dir / f"cnn_denoiser_snr_{_snr_to_fname_piece(snr_db)}dB.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "snr_db": float(snr_db),
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "config": asdict(cfg),
                    "normalize_mode": cfg.normalize_mode,
                    "normalize_stats": norm_stats,
                },
                ckpt,
            )
        else:
            patience += 1

        if patience >= cfg.early_stop_patience:
            print(f"[snr={snr_db:g} dB] early stop @ epoch {epoch+1} (best epoch {best_epoch+1}, best val {best_val:.6e})")
            break

    np.savez(
        logs_dir / "loss_history.npz",
        train=np.asarray(train_hist, dtype=np.float32),
        val=np.asarray(val_hist, dtype=np.float32),
    )

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_hist, label="train")
        plt.plot(val_hist, label="val")
        plt.xlabel("epoch")
        plt.ylabel("MSE loss")
        plt.title(f"CNN training (SNR={snr_db:g} dB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"loss_snr_{_snr_to_fname_piece(snr_db)}dB.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"[snr={snr_db:g} dB] Could not save loss plot: {e}")


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for snr_db in cfg.gen2_snr_db_list:
        print("\n" + "=" * 80)
        print(f"Training model for Gen2 SNR = {snr_db:g} dB")
        print("=" * 80)
        train_for_snr(cfg, snr_db, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
