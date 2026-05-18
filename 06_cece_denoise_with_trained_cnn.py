# 06_cece_denoise_with_trained_cnn.py
"""
Script 6/6 — Apply trained synthetic-data CNN to raw CECE data and plot 4×6 PSD grids.

PATCHED to use the trained/evaluated model artifacts on C: drive.

What this version does:
- Uses the latest run under C:/synouts by default
- Prefers the BEST evaluated checkpoint from:
    C:/synouts/<RUN_TAG>/eval_suite/tables/best_by_dataset.csv
- Falls back to:
    models/final_model_index.json
    newest cnn_final__noisey_*.pth
    newest *.pth in models/
- Rebuilds the exact CNNDenoiser1D architecture from ckpt["config"]
- Applies training-matched normalization:
    * per_sample
    * global
    * none
- Computes PSDs from raw CECE channel time series
- Denoises each PSD
- Plots 4×6 grids
- Optionally saves CSVs, figures, and metadata

Edit ONLY the USER SETTINGS block if needed.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import csd

import torch
import torch.nn as nn


# =========================
# ===== USER SETTINGS =====
# =========================

@dataclass
class RunConfig:
    # ---- CECE raw data ----
    folder: str = r"./43665data2"       # folder with per-channel .xlsx files
    pattern: str = "43665_ch*.xlsx"
    time_col: str = "time"
    value_col: Optional[str] = "voltage"   # if None, picks a voltage-like column
    expected_n: int = 24

    # ---- PSD params ----
    fs: Optional[float] = None            # if None, inferred from TIME_COL
    nperseg: int = 4096
    noverlap: int = 2048
    window: str = "hann"
    nfft: Optional[int] = None
    detrend: str = "constant"             # "constant" | "linear" | "none"

    # Optional Hann PSD gain correction
    # ONLY keep True if your synthetic/training PSDs used the same correction.
    apply_hann_gain_corr: bool = True
    hann_psd_gain: float = 8.0 / 3.0

    # ---- Model selection from C: drive ----
    synouts_root: str = r"C:/synouts"
    run_tag: str = "latest"               # "latest" or exact synouts run folder name

    # Optional manual override:
    # if not None, this exact checkpoint is used.
    ckpt_path: Optional[str] = None

    # Prefer evaluated best checkpoint if available
    prefer_best_evaluated_ckpt: bool = True

    # If best_by_dataset has multiple rows, choose the first row after sorting by mse_pred_vs_target
    # or choose a specific dataset name here.
    preferred_eval_dataset: Optional[str] = None

    # ---- Plotting ----
    xlim_hz: Optional[Tuple[float, float]] = (0, 150e3)
    title_prefix: str = ""
    figsize: Tuple[float, float] = (18, 10)

    # ---- Output ----
    save_csv: bool = True
    save_figs: bool = True
    fig_dpi: int = 200

    # Default output goes inside the selected run folder:
    #   C:/synouts/<RUN_TAG>/cece_outputs/
    output_subdir: str = "cece_outputs"

    # ---- Device ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Numeric stability
    eps_std: float = 1e-12


CFG = RunConfig()


# ==========================================
# =============== Utilities ================
# ==========================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def choose_device(which: str) -> torch.device:
    w = (which or "auto").lower()
    if w == "cpu":
        return torch.device("cpu")
    if w == "cuda":
        if not torch.cuda.is_available():
            print("[warn] CUDA requested but not available; using CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_latest_dir(root: Path) -> Path:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No directories found under: {root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_run_dir(synouts_root: Path, run_tag: str) -> Path:
    if str(run_tag).strip().lower() == "latest":
        return resolve_latest_dir(synouts_root)

    p = (synouts_root / run_tag).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Requested run_tag does not exist: {p}")
    return p


def parse_tag_from_ckpt_name(name: str) -> str:
    return Path(name).stem.replace(".", "p")


# ==========================================
# ===== Data reading helpers (from you) =====
# ==========================================

def read_channel_xlsx(xlsx_path, time_col="time", value_col=None, n=None):
    df = pd.read_excel(xlsx_path)
    if time_col not in df.columns:
        raise ValueError(f"{xlsx_path}: no time column '{time_col}'. Columns: {df.columns.tolist()}")

    if value_col is None:
        candidates = [c for c in df.columns if c.lower() != time_col.lower()]
        if not candidates:
            raise ValueError(f"{xlsx_path}: no data columns besides '{time_col}'.")
        pref = [c for c in candidates if any(k in c.lower() for k in ["v", "volt", "ch", "signal"])]
        use = pref[0] if pref else candidates[0]
    else:
        if value_col not in df.columns:
            raise ValueError(f"{xlsx_path}: value_col='{value_col}' not found. Columns: {df.columns.tolist()}")
        use = value_col

    t = df[time_col].to_numpy()
    v = df[use].to_numpy()

    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.isfinite(t) & np.isfinite(v)
    t, v = t[m], v[m]

    if n is not None:
        t, v = t[:n], v[:n]
    return t, v, use


def estimate_fs_from_time(t):
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        raise ValueError("Not enough valid time samples to infer fs.")
    dt = np.median(np.diff(np.sort(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not infer fs from time column (nonpositive/NaN dt).")
    return 1.0 / dt


def _channel_index_from_name(stem: str) -> int:
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"No digits found in channel name '{stem}'. Name must contain a channel number.")
    return int(nums[-1])


def load_24_channels(folder, pattern, time_col, value_col, expected_n):
    paths = list(Path(folder).glob(pattern))
    if expected_n is not None and len(paths) != expected_n:
        raise ValueError(f"Expected {expected_n} files, found {len(paths)} in '{folder}' pattern '{pattern}'")

    paths.sort(key=lambda p: _channel_index_from_name(p.stem))
    names = [p.stem for p in paths]

    channels = {}
    picked_cols = {}
    t_ref = None
    for p in paths:
        t, v, used = read_channel_xlsx(p, time_col=time_col, value_col=value_col)
        channels[p.stem] = v
        picked_cols[p.stem] = used
        if t_ref is None:
            t_ref = t
    return names, channels, picked_cols, t_ref, paths


# ==========================================
# ===== PSD computation (Welch via CSD) =====
# ==========================================

def _detrend_arg(detrend):
    return False if detrend == "none" else detrend


def autopower_psd(
    x,
    fs,
    nperseg,
    noverlap,
    window,
    nfft,
    detrend,
    apply_hann_gain_corr=True,
    hann_psd_gain=8.0 / 3.0,
):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max(8, int(nperseg)):
        raise ValueError(f"Signal too short for PSD: len={x.size}, nperseg={nperseg}")

    f, Sxx = csd(
        x, x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        nfft=nfft,
        detrend=_detrend_arg(detrend),
        scaling="density",
    )
    Sxx = np.real(Sxx).astype(np.float64)

    if apply_hann_gain_corr and str(window).lower() in ("hann", "hanning"):
        Sxx = Sxx * float(hann_psd_gain)

    Sxx = np.maximum(Sxx, 0.0)
    return f, Sxx


# ==========================================
# ============== Plotting grids ============
# ==========================================

def _grid_axes(nrows=4, ncols=6, figsize=(18, 10), sharex=True, sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    return fig, axes


def plot_grid_psd(psd_dict, order, xlim_hz=None, title_prefix="", suptitle="PSD", figsize=(18, 10)):
    fig, axes = _grid_axes(figsize=figsize)
    for idx, name in enumerate(order):
        r, c = divmod(idx, 6)
        ax = axes[r, c]
        f, Sxx = psd_dict[name]
        ax.plot(f, Sxx)
        if xlim_hz:
            ax.set_xlim(*xlim_hz)
        ax.grid(True, which="both")
        ax.set_title(f"{title_prefix}{name}", fontsize=9)
        if r == 3:
            ax.set_xlabel("Hz")
        if c == 0:
            ax.set_ylabel("PSD [V²/Hz]")
    fig.suptitle(suptitle, y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ==========================================
# ============== TRAINING MODEL ============
# (Exact match to script 02)
# ==========================================

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


# ==========================================
# ========= Checkpoint resolution ==========
# ==========================================

def try_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ckpt_from_best_eval(run_dir: Path, preferred_dataset: Optional[str] = None) -> Optional[Path]:
    table_path = run_dir / "eval_suite" / "tables" / "best_by_dataset.csv"
    if not table_path.exists():
        return None

    df = pd.read_csv(table_path)
    if df.empty or "ckpt" not in df.columns:
        return None

    if preferred_dataset is not None and "dataset" in df.columns:
        df2 = df[df["dataset"].astype(str) == str(preferred_dataset)]
        if not df2.empty:
            df = df2

    if "mse_pred_vs_target" in df.columns:
        df = df.sort_values("mse_pred_vs_target", ascending=True)

    ckpt_str = str(df.iloc[0]["ckpt"])
    ckpt_path = Path(ckpt_str)
    if ckpt_path.exists():
        return ckpt_path.resolve()

    return None


def resolve_ckpt_from_final_index(run_dir: Path) -> Optional[Path]:
    index_path = run_dir / "models" / "final_model_index.json"
    idx = try_load_json(index_path)
    if idx is None:
        return None

    final_models = idx.get("final_models", {})
    if not isinstance(final_models, dict) or len(final_models) == 0:
        return None

    # pick the first model in sorted key order for reproducibility
    for k in sorted(final_models.keys()):
        p = Path(str(final_models[k]))
        if p.exists():
            return p.resolve()

    return None


def resolve_newest_matching_ckpt(models_dir: Path, pattern: str) -> Optional[Path]:
    cands = list(models_dir.glob(pattern))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].resolve()


def resolve_checkpoint(cfg: RunConfig, run_dir: Path) -> Path:
    # Manual override always wins
    if cfg.ckpt_path is not None and str(cfg.ckpt_path).strip():
        p = Path(cfg.ckpt_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manual ckpt_path does not exist: {p}")
        return p

    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # 1) Best evaluated checkpoint
    if cfg.prefer_best_evaluated_ckpt:
        p = resolve_ckpt_from_best_eval(run_dir, preferred_dataset=cfg.preferred_eval_dataset)
        if p is not None:
            return p

    # 2) final_model_index.json
    p = resolve_ckpt_from_final_index(run_dir)
    if p is not None:
        return p

    # 3) newest final checkpoint
    p = resolve_newest_matching_ckpt(models_dir, "cnn_final__noisey_*.pth")
    if p is not None:
        return p

    # 4) newest any checkpoint
    p = resolve_newest_matching_ckpt(models_dir, "*.pth")
    if p is not None:
        return p

    raise FileNotFoundError(f"No usable checkpoint found in {models_dir}")


def load_ckpt(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(
            f"Expected checkpoint dict with key 'model_state'. Got incompatible file: {ckpt_path}"
        )

    cfg = ckpt.get("config", {}) or {}
    state = ckpt["model_state"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    normalize_mode = str(ckpt.get("normalize_mode", "none")).lower().strip()
    normalize_stats = ckpt.get("normalize_stats", {}) or {}

    return ckpt, cfg, state, normalize_mode, normalize_stats


def build_trained_model_from_ckpt_cfg(cfg: dict) -> CNNDenoiser1D:
    base_channels = int(cfg.get("base_channels", 32))
    kernel_size = int(cfg.get("kernel_size", 7))
    dropout = float(cfg.get("dropout", 0.0))
    return CNNDenoiser1D(
        base_channels=base_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )


# ==========================================
# ========= Normalization helpers ==========
# ==========================================

def normalize_per_sample(x: np.ndarray, eps: float):
    mu = float(np.mean(x))
    sig = float(np.std(x) + eps)
    return (x - mu) / sig, mu, sig


def normalize_global(x: np.ndarray, stats: Dict[str, float], eps: float):
    if "mu" not in stats or "sigma" not in stats:
        raise ValueError("Checkpoint normalize_mode='global' but mu/sigma missing from normalize_stats.")
    mu = float(stats["mu"])
    sig = float(stats["sigma"]) + eps
    return (x - mu) / sig, mu, sig


@torch.no_grad()
def denoise_psd_with_training_norm(
    model: nn.Module,
    psd: np.ndarray,
    device: torch.device,
    normalize_mode: str,
    normalize_stats: Optional[Dict[str, float]],
    eps: float,
):
    """
    Apply the SAME normalization convention as training and invert the prediction.
    """
    psd = np.asarray(psd, dtype=np.float64)
    psd = np.maximum(psd, 0.0)

    if normalize_mode == "per_sample":
        x_norm, mu, sig = normalize_per_sample(psd, eps=eps)
        xb = torch.from_numpy(x_norm.astype(np.float32))[None, None, :]
        xb = xb.to(device)
        y_norm = model(xb).detach().cpu().numpy()[0, 0, :].astype(np.float64)
        y = y_norm * sig + mu
        return np.maximum(y, 0.0)

    if normalize_mode == "global":
        x_norm, mu, sig = normalize_global(psd, stats=(normalize_stats or {}), eps=eps)
        xb = torch.from_numpy(x_norm.astype(np.float32))[None, None, :]
        xb = xb.to(device)
        y_norm = model(xb).detach().cpu().numpy()[0, 0, :].astype(np.float64)
        y = y_norm * sig + mu
        return np.maximum(y, 0.0)

    if normalize_mode in ("none", "raw"):
        xb = torch.from_numpy(psd.astype(np.float32))[None, None, :]
        xb = xb.to(device)
        y = model(xb).detach().cpu().numpy()[0, 0, :].astype(np.float64)
        return np.maximum(y, 0.0)

    raise ValueError(
        f"Unsupported normalize_mode='{normalize_mode}'. "
        f"Supported: per_sample, global, none."
    )


# ==========================================
# ============== Inference =================
# ==========================================

@torch.no_grad()
def denoise_all_channels(
    model: nn.Module,
    channels: Dict[str, np.ndarray],
    fs: float,
    normalize_mode: str,
    normalize_stats: Dict[str, float],
    cfg: RunConfig,
    device: torch.device,
):
    names = list(channels.keys())

    raw_psd = {}
    den_psd = {}

    for nm in names:
        f, Sxx = autopower_psd(
            channels[nm],
            fs=fs,
            nperseg=cfg.nperseg,
            noverlap=cfg.noverlap,
            window=cfg.window,
            nfft=cfg.nfft,
            detrend=cfg.detrend,
            apply_hann_gain_corr=cfg.apply_hann_gain_corr,
            hann_psd_gain=cfg.hann_psd_gain,
        )

        y_den = denoise_psd_with_training_norm(
            model=model,
            psd=Sxx,
            device=device,
            normalize_mode=normalize_mode,
            normalize_stats=normalize_stats,
            eps=cfg.eps_std,
        )

        raw_psd[nm] = (f, Sxx)
        den_psd[nm] = (f, y_den)

    return raw_psd, den_psd


def save_psd_csv(outdir: Path, psd_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], tag: str):
    ensure_dir(outdir)
    for name, (f, S) in psd_dict.items():
        pd.DataFrame({"f": f, f"{tag}_psd": S}).to_csv(outdir / f"{name}__{tag}_psd.csv", index=False)


def summarize_psd_dict(psd_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for name, (f, s) in psd_dict.items():
        rows.append(
            {
                "channel": name,
                "n_freq": int(len(f)),
                "psd_min": float(np.min(s)),
                "psd_max": float(np.max(s)),
                "psd_mean": float(np.mean(s)),
                "psd_std": float(np.std(s)),
                "psd_integral_trapz": float(np.trapz(s, f)),
            }
        )
    return pd.DataFrame(rows)


def main():
    device = choose_device(CFG.device)

    synouts_root = Path(CFG.synouts_root).resolve()
    run_dir = resolve_run_dir(synouts_root, CFG.run_tag)
    ckpt_path = resolve_checkpoint(CFG, run_dir)

    outdir = run_dir / CFG.output_subdir
    ensure_dir(outdir)

    print(f"[06] Using device: {device}")
    print(f"[06] Using synouts run folder: {run_dir}")
    print(f"[06] Using checkpoint: {ckpt_path}")

    # Load channels
    order, channels, picked_cols, t_ref, paths = load_24_channels(
        CFG.folder,
        CFG.pattern,
        CFG.time_col,
        CFG.value_col,
        CFG.expected_n,
    )

    # Sampling rate
    fs = CFG.fs if CFG.fs is not None else estimate_fs_from_time(t_ref)
    print(f"[06] Using fs = {fs:.6g} Hz")

    # Load checkpoint + build model
    ckpt, train_cfg, state, normalize_mode, normalize_stats = load_ckpt(ckpt_path, device)
    print(f"[06] Loaded checkpoint metadata:")
    print(f"     run_id         = {ckpt.get('run_id', None)}")
    print(f"     run_tag        = {ckpt.get('run_tag', None)}")
    print(f"     noise_y_bucket = {ckpt.get('noise_y_bucket', None)}")
    print(f"     stage_name     = {ckpt.get('stage_name', ckpt.get('final_stage', None))}")
    print(f"     best_val       = {ckpt.get('best_val', None)}")
    print(f"     best_epoch     = {ckpt.get('best_epoch', None)}")
    print(f"     normalize_mode = {normalize_mode}")

    model = build_trained_model_from_ckpt_cfg(train_cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Denoise
    raw_psd, den_psd = denoise_all_channels(
        model=model,
        channels=channels,
        fs=fs,
        normalize_mode=normalize_mode,
        normalize_stats=normalize_stats,
        cfg=CFG,
        device=device,
    )

    # Save outputs
    if CFG.save_csv:
        save_psd_csv(outdir / "raw", raw_psd, tag="raw")
        save_psd_csv(outdir / "denoised", den_psd, tag="denoised")

        raw_summary = summarize_psd_dict(raw_psd)
        den_summary = summarize_psd_dict(den_psd)
        raw_summary.to_csv(outdir / "raw_psd_summary.csv", index=False)
        den_summary.to_csv(outdir / "denoised_psd_summary.csv", index=False)

        meta = {
            "synouts_root": str(synouts_root),
            "run_dir": str(run_dir),
            "ckpt_path": str(ckpt_path),
            "ckpt_name": ckpt_path.name,
            "run_id": ckpt.get("run_id", None),
            "run_tag": ckpt.get("run_tag", None),
            "noise_y_bucket": ckpt.get("noise_y_bucket", None),
            "stage_name": ckpt.get("stage_name", ckpt.get("final_stage", None)),
            "best_val": ckpt.get("best_val", None),
            "best_epoch": ckpt.get("best_epoch", None),
            "normalize_mode": normalize_mode,
            "normalize_stats": normalize_stats,
            "picked_value_columns": picked_cols,
            "cece_files": [str(p.resolve()) for p in paths],
            "psd_params": {
                "fs": float(fs),
                "nperseg": int(CFG.nperseg),
                "noverlap": int(CFG.noverlap),
                "window": str(CFG.window),
                "nfft": None if CFG.nfft is None else int(CFG.nfft),
                "detrend": str(CFG.detrend),
                "apply_hann_gain_corr": bool(CFG.apply_hann_gain_corr),
                "hann_psd_gain": float(CFG.hann_psd_gain),
            },
            "train_config": train_cfg,
        }
        with open(outdir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Plot
    fig1 = plot_grid_psd(
        raw_psd,
        order,
        xlim_hz=CFG.xlim_hz,
        title_prefix=CFG.title_prefix,
        suptitle="Raw / Observed PSD",
        figsize=CFG.figsize,
    )
    fig2 = plot_grid_psd(
        den_psd,
        order,
        xlim_hz=CFG.xlim_hz,
        title_prefix=CFG.title_prefix,
        suptitle="CNN Denoised PSD",
        figsize=CFG.figsize,
    )

    if CFG.save_figs:
        fig1.savefig(outdir / "psd_raw_4x6.png", dpi=CFG.fig_dpi)
        fig2.savefig(outdir / "psd_denoised_4x6.png", dpi=CFG.fig_dpi)

    print(f"[06] Outputs saved under: {outdir}")
    plt.show()


if __name__ == "__main__":
    main()