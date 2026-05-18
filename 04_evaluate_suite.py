# 04_evaluate_suite.py
"""
Evaluate CNN PSD denoiser checkpoints on one or more eval .npz datasets.

Matches pipeline script 03_infer_predict_psd.py assumptions:
- Checkpoints: outputs/models/*.pth (dict with key 'model_state')
- Eval datasets: synthetic_datasets/datasets/*.npz (must contain f, Y_obs, Y_tilde; params optional)
- Applies SAME normalization/denormalization as training/inference.
- Computes baseline (Y_obs vs Y_tilde) and model (Y_pred vs Y_tilde) metrics, plus improvements.
- Saves summary tables + plots under outputs/eval_suite/

No CLI needed: edit CONFIG and run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Config (edit me)
# ----------------------------

@dataclass
class EvalSuiteConfig:
    # Where training artifacts live (same as TrainConfig.out_root / InferConfig.out_root)
    out_root: str = "outputs"

    # Which eval datasets to evaluate (glob or explicit list)
    # Examples:
    #   "synthetic_datasets/datasets/eval*.npz"
    #   "synthetic_datasets/datasets/eval_mixed_snr.npz"
    eval_glob: str = "synthetic_datasets/datasets/eval*.npz"
    eval_paths: Tuple[str, ...] = ()  # if non-empty, overrides eval_glob

    # Which model(s) to run
    ckpt_paths: Tuple[str, ...] = ()  # if non-empty, uses those exact files
    ckpt_glob_pattern: str = "cnn_denoiser_snr_*dB.pth"  # searched in outputs/models/

    # Output subdir under out_root
    eval_subdir: str = "eval_suite"

    # Dataloader
    batch_size: int = 512
    num_workers: int = 0

    # Device
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # Plots
    n_plot: int = 18
    plot_seed: int = 0


# ----------------------------
# Model (must match script 02/03)
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
# Helpers (match script 03 behavior)
# ----------------------------

def append_text(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


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


def load_eval_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    for k in ("f", "Y_obs", "Y_tilde"):
        if k not in d:
            raise KeyError(f"{npz_path} missing '{k}'. Keys: {list(d.keys())}")
    return {
        "f": d["f"].astype(np.float64),
        "Y_obs": d["Y_obs"].astype(np.float32),
        "Y_tilde": d["Y_tilde"].astype(np.float32),
        "params": d["params"] if "params" in d.files else None,
    }


def normalize_arrays(
    X_noisy: np.ndarray,
    X_clean: np.ndarray,
    mode: str,
    eps: float,
    stats_in: Optional[Dict[str, float]] = None,
):
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
        mu = np.mean(X_noisy, axis=1, keepdims=True)
        sig = np.std(X_noisy, axis=1, keepdims=True) + eps
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


def parse_tag_from_ckpt_name(name: str) -> str:
    return Path(name).stem.replace(".", "p")


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
# Core evaluation for one ckpt
# ----------------------------

@torch.no_grad()
def predict_one_ckpt(
    ckpt_path: Path,
    cfg: EvalSuiteConfig,
    dset: Dict[str, np.ndarray],
    device: torch.device,
    out_dir: Path,
) -> Dict[str, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"{ckpt_path} has unexpected format. Expected dict with key 'model_state'.")

    train_cfg = ckpt.get("config", {})
    base_channels = int(train_cfg.get("base_channels", 32))
    kernel_size = int(train_cfg.get("kernel_size", 7))
    dropout = float(train_cfg.get("dropout", 0.0))
    eps = float(train_cfg.get("eps", 1e-12))

    normalize_mode = str(ckpt.get("normalize_mode", "none"))
    norm_stats = ckpt.get("normalize_stats", {})

    model = CNNDenoiser1D(base_channels=base_channels, kernel_size=kernel_size, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    f = dset["f"]
    y_obs = dset["Y_obs"]
    y_tilde = dset["Y_tilde"]

    # normalize like training
    y_obs_norm, y_tilde_norm, _ = normalize_arrays(
        y_obs,
        y_tilde,
        mode=normalize_mode,
        eps=eps,
        stats_in=(norm_stats if normalize_mode == "global" else None),
    )

    x = torch.from_numpy(y_obs_norm).float().unsqueeze(1)   # (N,1,F)
    y = torch.from_numpy(y_tilde_norm).float().unsqueeze(1) # (N,1,F)

    dl = DataLoader(
        TensorDataset(x, y),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    preds_norm: List[np.ndarray] = []
    for xb, _ in dl:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb)  # (B,1,F)
        preds_norm.append(pred.detach().cpu().numpy())

    y_pred_norm = np.concatenate(preds_norm, axis=0).squeeze(1)  # (N,F)
    y_pred = denormalize_pred(y_pred_norm, y_obs=y_obs, mode=normalize_mode, eps=eps, stats=norm_stats)

    metrics = compute_metrics(y_tilde=y_tilde, y_obs=y_obs, y_pred=y_pred)

    tag = parse_tag_from_ckpt_name(ckpt_path.name)
    ensure_dir(out_dir)

    # Save per-ckpt metrics json
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

    # Quicklook plot
    plot_overlays(
        out_png=out_dir / f"quicklook_overlays_{tag}.png",
        f=f,
        y_obs=y_obs,
        y_tilde=y_tilde,
        y_pred=y_pred,
        n_plot=cfg.n_plot,
        seed=cfg.plot_seed,
    )

    return {
        "tag": tag,
        "ckpt": str(ckpt_path),
        "snr_db": float(ckpt.get("snr_db", float("nan"))),
        "best_val": float(ckpt.get("best_val", float("nan"))),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        **metrics,
    }


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    cfg = EvalSuiteConfig()
    device = choose_device(cfg.device)
    print(f"Using device: {device}")

    # Your real layout:
    # .../masters work/outputs/models
    # .../masters work/denoising_pipeline_syn/04_evaluate_suite.py
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    out_root = (project_root / cfg.out_root).resolve()

    models_dir = out_root / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models dir not found: {models_dir}")

    # datasets
    if cfg.eval_paths:
        eval_paths = [(project_root / p).resolve() for p in cfg.eval_paths]
    else:
        eval_paths = sorted((project_root).glob(cfg.eval_glob))
        eval_paths = [p.resolve() for p in eval_paths]

    if not eval_paths:
        raise FileNotFoundError(
            f"No eval datasets found.\n"
            f"Project root: {project_root}\n"
            f"Tried eval_glob={cfg.eval_glob}"
        )

    # checkpoints
    if cfg.ckpt_paths:
        ckpts = [(project_root / p).resolve() for p in cfg.ckpt_paths]
    else:
        ckpts = sorted(models_dir.glob(cfg.ckpt_glob_pattern))

    if not ckpts:
        raise FileNotFoundError(
            "No checkpoints found.\n"
            f"Looked in: {models_dir}\n"
            f"With glob: {cfg.ckpt_glob_pattern}"
        )

    eval_root = out_root / cfg.eval_subdir
    ensure_dir(eval_root)
    ensure_dir(eval_root / "tables")

    print(f"Found {len(eval_paths)} eval dataset(s).")
    print(f"Found {len(ckpts)} checkpoint(s).")
    print(f"Outputs -> {eval_root}")

    # ----------------------------
    # Evaluation loop (+ debug logging)
    # ----------------------------
    all_rows: List[Dict[str, object]] = []
    err_log = eval_root / "errors.log"

    for eval_npz in eval_paths:
        dname = eval_npz.stem

        # Create dataset folder immediately so you can see progress
        d_out = eval_root / dname
        ensure_dir(d_out)

        # Load dataset (hard fail if this breaks)
        try:
            dset = load_eval_npz(eval_npz)
        except Exception as e:
            append_text(err_log, f"[DATASET LOAD FAIL] {eval_npz} :: {repr(e)}")
            raise

        for ckpt in ckpts:
            tag = parse_tag_from_ckpt_name(ckpt.name)
            out_dir = d_out / tag
            ensure_dir(out_dir)

            append_text(out_dir / "STARTED.txt", f"Evaluating ckpt={ckpt} on eval={eval_npz}")

            try:
                row = predict_one_ckpt(ckpt, cfg, dset, device, out_dir)
                row["dataset"] = dname
                row["eval_npz"] = str(eval_npz)
                all_rows.append(row)

                print(
                    f"[eval] {dname} | {tag} | mse_pred={row['mse_pred_vs_target']:.4e} "
                    f"(improve x{row['rmse_improvement_factor']:.2f})"
                )
                append_text(out_dir / "DONE.txt", "OK")

            except Exception as e:
                append_text(err_log, f"[EVAL FAIL] dataset={dname} ckpt={ckpt} :: {repr(e)}")
                append_text(out_dir / "FAILED.txt", repr(e))
                print(f"[warn] Failed: {dname} | {tag} -> {e}")
                continue

    if not all_rows:
        raise RuntimeError(f"All evaluations failed. See log: {err_log}")

    print(f"Collected {len(all_rows)} successful eval rows; writing tables…")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["dataset", "mse_pred_vs_target"]).reset_index(drop=True)

    tables_dir = eval_root / "tables"
    ensure_dir(tables_dir)

    df.to_csv(tables_dir / "results_by_ckpt.csv", index=False)

    best = df.sort_values(["dataset", "mse_pred_vs_target"]).groupby("dataset", as_index=False).head(1)
    best.to_csv(tables_dir / "best_by_dataset.csv", index=False)

    with open(eval_root / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "project_root": str(project_root),
                "out_root": str(out_root),
                "models_dir": str(models_dir),
                "n_ckpts": int(len(ckpts)),
                "n_datasets": int(len(eval_paths)),
                "eval_paths": [str(p) for p in eval_paths],
                "ckpts": [str(p) for p in ckpts],
                "tables": {
                    "results_by_ckpt": str(tables_dir / "results_by_ckpt.csv"),
                    "best_by_dataset": str(tables_dir / "best_by_dataset.csv"),
                },
                "error_log": str(err_log) if err_log.exists() else None,
            },
            f,
            indent=2,
        )

    print("Done.")


if __name__ == "__main__":
    main()
