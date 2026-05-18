# 03_infer_predict_psd.py
"""
Run inference with CNN denoiser trained by:
  02_train_cnn_generation.py

This script:
- Loads one (or many) trained checkpoint(s) from outputs/models/
- Loads an eval .npz produced by 01_synth_make_datasets.py
- Applies the SAME normalization used during training (none/global/per_sample)
- Predicts denoised PSD (Y_pred) from observed PSD (Y_obs)
- Saves:
    outputs/infer/<tag>/predictions_*.npz
    outputs/infer/<tag>/quicklook_*.png
    outputs/infer/<tag>/metrics_*.json

Dataset .npz (from script 01) must contain:
  - f       (F,)
  - Y_obs   (N,F)
  - Y_tilde (N,F)
Optional:
  - params  (N,...)  (stored through)

No CLI needed: edit CONFIG and run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Config (edit me)
# ----------------------------

@dataclass
class InferConfig:
    # Where training artifacts live (same as TrainConfig.out_root)
    out_root: str = "outputs"

    # Eval dataset produced by script 01
    # Examples:
    #   synthetic_datasets/datasets/eval_mixed_snr.npz
    #   synthetic_datasets/datasets/train_gen3_mix_50_50.npz
    eval_npz: str = "synthetic_datasets/datasets/eval_mixed_snr.npz"

    # Which model(s) to run:
    # - If ckpt_paths is non-empty, uses those exact files.
    # - Else, will infer over all checkpoints matching glob_pattern in outputs/models/
    ckpt_paths: Tuple[str, ...] = ()
    glob_pattern: str = "cnn_denoiser_snr_*dB.pth"

    # Output subdir under out_root
    infer_subdir: str = "infer"

    # Dataloader
    batch_size: int = 512
    num_workers: int = 0

    # Device
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # How many random samples to plot as overlays
    n_plot: int = 18
    plot_seed: int = 0

    # Save full prediction npz
    save_pred_npz: bool = True


# ----------------------------
# Model (must match script 02)
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
# Helpers
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def choose_device(which: str) -> torch.device:
    w = (which or "auto").lower()
    if w == "cpu":
        return torch.device("cpu")
    if w == "cuda":
        if not torch.cuda.is_available():
            print("[warn] device=cuda requested but CUDA not available; using cpu.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eval_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    d = np.load(npz_path, allow_pickle=True)
    for k in ("f", "Y_obs", "Y_tilde"):
        if k not in d:
            raise KeyError(f"{npz_path} missing '{k}'. Keys: {list(d.keys())}")

    f = d["f"].astype(np.float64)             # (F,)
    y_obs = d["Y_obs"].astype(np.float32)     # (N,F)
    y_tilde = d["Y_tilde"].astype(np.float32) # (N,F)
    params = d["params"] if "params" in d.files else None
    return f, y_obs, y_tilde, params


def normalize_arrays(
    X_noisy: np.ndarray,
    X_clean: np.ndarray,
    mode: str,
    eps: float,
    stats_in: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Must match script 02 behavior.
    If stats_in is provided, we use those stats (important for eval/infer).
    """
    stats: Dict[str, float] = {} if stats_in is None else dict(stats_in)

    if mode == "none":
        return X_noisy, X_clean, stats

    if mode == "global":
        if stats_in is None:
            mu = float(np.mean(X_noisy))
            sig = float(np.std(X_noisy) + eps)
            stats.update({"mu": mu, "sigma": sig})
        mu = float(stats["mu"])
        sig = float(stats["sigma"])
        return (X_noisy - mu) / sig, (X_clean - mu) / sig, stats

    if mode == "per_sample":
        # per-sample stats are computed from the input at inference time
        mu = np.mean(X_noisy, axis=1, keepdims=True)
        sig = np.std(X_noisy, axis=1, keepdims=True) + eps
        # store summary only (like training did)
        if stats_in is None:
            stats.update({"mu_mean": float(mu.mean()), "sigma_mean": float(sig.mean())})
        return (X_noisy - mu) / sig, (X_clean - mu) / sig, stats

    raise ValueError(f"Unknown normalize_mode='{mode}'.")


def denormalize_pred(
    y_pred_norm: np.ndarray,
    y_obs: np.ndarray,
    mode: str,
    eps: float,
    stats: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Inverse of training normalization to put predictions back in original PSD units.
    - none: identity
    - global: y = y_norm * sigma + mu
    - per_sample: y = y_norm * std(obs_i) + mean(obs_i)   (computed per sample from observed PSD)
    """
    if mode == "none":
        return y_pred_norm

    if mode == "global":
        if not stats or "mu" not in stats or "sigma" not in stats:
            raise ValueError("global denorm requested but stats missing mu/sigma.")
        mu = float(stats["mu"])
        sig = float(stats["sigma"])
        return y_pred_norm * sig + mu

    if mode == "per_sample":
        mu = np.mean(y_obs, axis=1, keepdims=True)
        sig = np.std(y_obs, axis=1, keepdims=True) + eps
        return y_pred_norm * sig + mu

    raise ValueError(f"Unknown normalize_mode='{mode}'.")


def parse_snr_tag_from_name(name: str) -> str:
    # "cnn_denoiser_snr_20dB.pth" or "cnn_denoiser_snr_7p5dB.pth"
    stem = Path(name).stem
    # make a safe label
    return stem.replace(".", "p")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def r2(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def compute_metrics(y_tilde: np.ndarray, y_obs: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse_obs = mse(y_tilde, y_obs)
    mse_pred = mse(y_tilde, y_pred)
    return {
        "mse_obs_vs_target": mse_obs,
        "mse_pred_vs_target": mse_pred,
        "mae_obs_vs_target": mae(y_tilde, y_obs),
        "mae_pred_vs_target": mae(y_tilde, y_pred),
        "r2_pred_vs_target": r2(y_tilde, y_pred),
        "rmse_improvement_factor": math.sqrt(mse_obs) / max(math.sqrt(mse_pred), 1e-30),
    }


def plot_overlays(
    out_png: Path,
    f: np.ndarray,
    y_obs: np.ndarray,
    y_tilde: np.ndarray,
    y_pred: np.ndarray,
    n_plot: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    N = y_obs.shape[0]
    n_plot = int(min(max(n_plot, 1), N))
    idx = rng.choice(N, size=n_plot, replace=False)

    ncols = int(math.ceil(math.sqrt(n_plot)))
    nrows = int(math.ceil(n_plot / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), squeeze=False)
    axes = axes.ravel()

    for i, k in enumerate(idx):
        ax = axes[i]
        ax.plot(f, y_obs[k], label="Y_obs (raw)")
        ax.plot(f, y_pred[k], label="Y_pred (denoised)")
        ax.plot(f, y_tilde[k], label="Y_tilde (target)")
        ax.set_title(f"sample {k}")
        ax.set_xlabel("f")
        ax.set_ylabel("PSD")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=9)

    for j in range(len(idx), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------
# Inference
# ----------------------------

@torch.no_grad()
def run_infer_one_ckpt(
    ckpt_path: Path,
    cfg: InferConfig,
    f: np.ndarray,
    y_obs: np.ndarray,
    y_tilde: np.ndarray,
    params: Optional[np.ndarray],
    device: torch.device,
    out_dir: Path,
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"{ckpt_path} has unexpected format. Expected dict with key 'model_state'.")

    # Model hyperparams (must match training)
    train_cfg = ckpt.get("config", {})
    base_channels = int(train_cfg.get("base_channels", 32))
    kernel_size = int(train_cfg.get("kernel_size", 7))
    dropout = float(train_cfg.get("dropout", 0.0))

    normalize_mode = str(ckpt.get("normalize_mode", "none"))
    eps = float(train_cfg.get("eps", 1e-12))
    norm_stats = ckpt.get("normalize_stats", {})

    model = CNNDenoiser1D(base_channels=base_channels, kernel_size=kernel_size, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Normalize eval data (same as training)
    y_obs_norm, y_tilde_norm, _ = normalize_arrays(
        y_obs, y_tilde, mode=normalize_mode, eps=eps, stats_in=(norm_stats if normalize_mode == "global" else None)
    )

    x = torch.from_numpy(y_obs_norm).float().unsqueeze(1)  # (N,1,F)
    y = torch.from_numpy(y_tilde_norm).float().unsqueeze(1)

    dl = DataLoader(
        TensorDataset(x, y),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    preds_norm: List[np.ndarray] = []
    for xb, _yb in dl:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb)  # (B,1,F)
        preds_norm.append(pred.detach().cpu().numpy())

    y_pred_norm = np.concatenate(preds_norm, axis=0).squeeze(1)  # (N,F)

    # Denormalize back to original PSD units
    y_pred = denormalize_pred(y_pred_norm, y_obs=y_obs, mode=normalize_mode, eps=eps, stats=norm_stats)

    # Metrics in original units
    metrics = compute_metrics(y_tilde=y_tilde, y_obs=y_obs, y_pred=y_pred)

    tag = parse_snr_tag_from_name(ckpt_path.name)
    ensure_dir(out_dir)

    # Save metrics
    with open(out_dir / f"metrics_{tag}.json", "w", encoding="utf-8") as f_out:
        json.dump(
            {
                "ckpt": str(ckpt_path),
                "tag": tag,
                "snr_db": float(ckpt.get("snr_db", float("nan"))),
                "best_val": float(ckpt.get("best_val", float("nan"))),
                "best_epoch": int(ckpt.get("best_epoch", -1)),
                "normalize_mode": normalize_mode,
                "normalize_stats": norm_stats,
                "metrics": metrics,
            },
            f_out,
            indent=2,
        )

    # Save predictions
    if cfg.save_pred_npz:
        np.savez(
            out_dir / f"predictions_{tag}.npz",
            f=f,
            Y_obs=y_obs,
            Y_tilde=y_tilde,
            Y_pred=y_pred,
            params=params if params is not None else None,
            meta={
                "ckpt": str(ckpt_path),
                "tag": tag,
                "normalize_mode": normalize_mode,
                "normalize_stats": norm_stats,
                "snr_db": float(ckpt.get("snr_db", float("nan"))),
                "best_val": float(ckpt.get("best_val", float("nan"))),
                "best_epoch": int(ckpt.get("best_epoch", -1)),
                "metrics": metrics,
            },
        )

    # Quicklook overlays
    plot_overlays(
        out_png=out_dir / f"quicklook_overlays_{tag}.png",
        f=f,
        y_obs=y_obs,
        y_tilde=y_tilde,
        y_pred=y_pred,
        n_plot=cfg.n_plot,
        seed=cfg.plot_seed,
    )

    print(f"[infer] {tag}: mse_pred_vs_target={metrics['mse_pred_vs_target']:.4e} "
          f"(improve x{metrics['rmse_improvement_factor']:.2f}) -> {out_dir}")


def main() -> None:
    cfg = InferConfig()
    device = choose_device(cfg.device)
    print(f"Using device: {device}")

    out_root = Path(cfg.out_root).resolve()
    models_dir = out_root / "models"

    eval_npz = Path(cfg.eval_npz).resolve()
    if not eval_npz.exists():
        raise FileNotFoundError(f"Eval npz not found: {eval_npz}")

    f, y_obs, y_tilde, params = load_eval_npz(eval_npz)
    if y_obs.ndim != 2 or y_tilde.ndim != 2:
        raise ValueError(f"Expected (N,F) arrays; got Y_obs {y_obs.shape}, Y_tilde {y_tilde.shape}")
    if y_obs.shape != y_tilde.shape:
        raise ValueError(f"Shape mismatch: Y_obs {y_obs.shape} vs Y_tilde {y_tilde.shape}")
    if f.shape[0] != y_obs.shape[1]:
        raise ValueError(f"f length {f.shape[0]} != F {y_obs.shape[1]}")

    # Select checkpoints
    ckpts: List[Path] = []
    if cfg.ckpt_paths:
        ckpts = [Path(p).resolve() for p in cfg.ckpt_paths]
    else:
        if not models_dir.exists():
            raise FileNotFoundError(f"Models dir not found: {models_dir}")
        ckpts = sorted(models_dir.glob(cfg.glob_pattern))

    if not ckpts:
        raise FileNotFoundError(
            "No checkpoints found.\n"
            f"Looked in: {models_dir}\n"
            f"With glob: {cfg.glob_pattern}\n"
            f"Or ckpt_paths: {cfg.ckpt_paths}"
        )

    infer_dir = out_root / cfg.infer_subdir
    ensure_dir(infer_dir)

    print(f"Eval dataset: {eval_npz}")
    print(f"Found {len(ckpts)} checkpoint(s).")

    # Run each checkpoint
    for ckpt_path in ckpts:
        tag = parse_snr_tag_from_name(ckpt_path.name)
        out_dir = infer_dir / tag
        run_infer_one_ckpt(
            ckpt_path=ckpt_path,
            cfg=cfg,
            f=f,
            y_obs=y_obs,
            y_tilde=y_tilde,
            params=params,
            device=device,
            out_dir=out_dir,
        )

    print("Done.")


if __name__ == "__main__":
    main()
