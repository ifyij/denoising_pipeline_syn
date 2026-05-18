from __future__ import annotations

"""
Script 7/7 — Compute CECE cross-power spectra for 24 pair slots, denoise each
cross-power magnitude with the trained synthetic-data CNN, and plot/save results.

PATCHED for compatibility with both the older PSD CNN checkpoints and the newer
"red-target" log-PSD residual checkpoints.

Key fixes:
1) Robust model reconstruction from checkpoint state_dict.
   - Handles the older body layout and the newer expanded layout.
   - Avoids state_dict size mismatch errors like body.5 / body.7 keys.

2) Robust checkpoint metadata parsing.
   - Supports config being nested or partially missing.
   - Detects log-PSD inference from checkpoint/config when present.

3) Robust denoising path.
   - Supports linear-output checkpoints.
   - Supports log-PSD residual checkpoints that predict a correction in log space.
"""

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
    folder: str = r"./43665data2"
    pattern: str = "43665_ch*.xlsx"
    time_col: str = "time"
    value_col: Optional[str] = "voltage"
    expected_n: int = 24

    fs: Optional[float] = None
    nperseg: int = 4096
    noverlap: int = 2048
    window: str = "hann"
    nfft: Optional[int] = None
    detrend: str = "constant"

    apply_hann_gain_corr: bool = True
    hann_psd_gain: float = 8.0 / 3.0

    pair_mode: str = "neighbor_row_repeat"

    synouts_root: str = r"C:/synouts"
    run_tag: str = "latest"
    ckpt_path: Optional[str] = None
    prefer_best_evaluated_ckpt: bool = True
    preferred_eval_dataset: Optional[str] = None

    xlim_hz: Optional[Tuple[float, float]] = (0, 150e3)
    title_prefix: str = ""
    figsize_grid: Tuple[float, float] = (18, 10)
    figsize_pair: Tuple[float, float] = (9, 5)
    save_loglin_pair_figs: bool = True

    # Physical plausibility guardrails for real CECE spectra. These do not
    # create information; they prevent a checkpoint from winning by collapsing
    # the whole spectrum into an unrealistically flat floor.
    apply_physical_guardrails: bool = True
    floor_est_freq_frac: float = 0.20
    peak_band_rel_height: float = 0.25
    min_floor_reduction_db: float = 1.0
    max_integral_change_frac: float = 0.35
    max_peak_change_frac: float = 0.35

    save_csv: bool = True
    save_figs: bool = True
    fig_dpi: int = 200
    output_subdir: str = "cece_cross_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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


def _get_first_present(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


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

    channels: Dict[str, np.ndarray] = {}
    picked_cols: Dict[str, str] = {}
    t_ref = None
    for p in paths:
        t, v, used = read_channel_xlsx(p, time_col=time_col, value_col=value_col)
        channels[p.stem] = v
        picked_cols[p.stem] = used
        if t_ref is None:
            t_ref = t
    return names, channels, picked_cols, t_ref, paths


# ==========================================
# ===== Spectral computation helpers =======
# ==========================================

def _detrend_arg(detrend):
    return False if detrend == "none" else detrend


def autopower_psd(x, fs, nperseg, noverlap, window, nfft, detrend, apply_hann_gain_corr=True, hann_psd_gain=8.0 / 3.0):
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

    return f, np.maximum(Sxx, 0.0)


def crosspower_csd(x, y, fs, nperseg, noverlap, window, nfft, detrend, apply_hann_gain_corr=True, hann_psd_gain=8.0 / 3.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if min(x.size, y.size) < max(8, int(nperseg)):
        raise ValueError(f"Signals too short for cross PSD: len(x)={x.size}, len(y)={y.size}, nperseg={nperseg}")

    f, Sxy = csd(
        x, y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        nfft=nfft,
        detrend=_detrend_arg(detrend),
        scaling="density",
    )
    Sxy = Sxy.astype(np.complex128)

    if apply_hann_gain_corr and str(window).lower() in ("hann", "hanning"):
        Sxy = Sxy * float(hann_psd_gain)

    return f, Sxy


# ==========================================
# ============== Pair selection ============
# ==========================================

def build_pair_slots(order: List[str], mode: str = "neighbor_row_repeat") -> List[Tuple[str, str]]:
    if mode != "neighbor_row_repeat":
        raise ValueError(f"Unsupported pair_mode='{mode}'")
    if len(order) != 24:
        raise ValueError(f"Expected 24 ordered channels, got {len(order)}")

    pairs: List[Tuple[str, str]] = []
    for row in range(4):
        base = row * 6
        row_names = order[base:base + 6]
        for c in range(6):
            if c < 5:
                a, b = row_names[c], row_names[c + 1]
            else:
                a, b = row_names[4], row_names[5]
            pairs.append((a, b))
    return pairs


def pair_label(a: str, b: str) -> str:
    return f"{a}__x__{b}"


# ==========================================
# ============== Plotting ==================
# ==========================================

def _grid_axes(nrows=4, ncols=6, figsize=(18, 10), sharex=True, sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    return fig, axes


def plot_grid_pair_metric(data_by_slot: List[Tuple[str, np.ndarray, np.ndarray]], xlim_hz=None, title_prefix="", suptitle="", ylabel="", figsize=(18, 10)):
    fig, axes = _grid_axes(figsize=figsize)
    for idx, (label, f, y) in enumerate(data_by_slot):
        r, c = divmod(idx, 6)
        ax = axes[r, c]
        ax.plot(f, y)
        if xlim_hz:
            ax.set_xlim(*xlim_hz)
        ax.grid(True, which="both")
        ax.set_title(f"{title_prefix}{label}", fontsize=9)
        if r == 3:
            ax.set_xlabel("Hz")
        if c == 0:
            ax.set_ylabel(ylabel)
    fig.suptitle(suptitle, y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_individual_pair_comparison(label: str, f: np.ndarray, auto_ref: np.ndarray, cross_mag: np.ndarray, den_mag: np.ndarray, xlim_hz=None, figsize=(9, 5), yscale: str = "linear", metrics: Optional[Dict[str, float]] = None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(f, auto_ref, label="sqrt(Sxx*Syy)")
    ax.plot(f, cross_mag, label="|Sxy|")
    ax.plot(f, den_mag, label="CNN denoised |Sxy|")
    if xlim_hz:
        ax.set_xlim(*xlim_hz)
    if str(yscale).lower() == "log":
        ax.set_yscale("log")
    ax.set_xlabel("Hz")
    ax.set_ylabel("PSD / Cross-power magnitude")
    title = label if str(yscale).lower() != "log" else f"{label} (log-lin)"
    if metrics:
        title += (
            f" | floor {metrics.get('floor_reduction_db', float('nan')):.1f} dB, "
            f"area {metrics.get('integral_ratio', float('nan')):.2f}, "
            f"peak {metrics.get('peak_ratio', float('nan')):.2f}"
        )
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    return fig


# ==========================================
# ============== MODEL DEFNS ===============
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


class CNNDenoiser1DLegacy(nn.Module):
    """Older architecture used by earlier training scripts."""
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


class CNNDenoiser1DExpanded(nn.Module):
    """Newer expanded architecture used by the red-target trainer."""
    def __init__(self, base_channels: int = 32, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        k = kernel_size
        pad = k // 2
        c = base_channels
        c_mid = max(1, c // 2)
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock1D(c, k, dropout=dropout),
            ResidualBlock1D(c, k, dropout=dropout),
            nn.Conv1d(c, c, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
            ResidualBlock1D(c, k, dropout=dropout),
            nn.Conv1d(c, c_mid, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
            ResidualBlock1D(c_mid, k, dropout=dropout),
        )
        self.head = nn.Conv1d(c_mid, 1, kernel_size=k, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.body(h)
        return self.head(h)


# ==========================================
# ========= Checkpoint resolution ===========
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
    elif "mse_pred" in df.columns:
        df = df.sort_values("mse_pred", ascending=True)

    ckpt_path = Path(str(df.iloc[0]["ckpt"]))
    return ckpt_path.resolve() if ckpt_path.exists() else None


def resolve_ckpt_from_final_index(run_dir: Path) -> Optional[Path]:
    index_path = run_dir / "models" / "final_model_index.json"
    idx = try_load_json(index_path)
    if idx is None:
        return None

    final_models = idx.get("final_models", {})
    if not isinstance(final_models, dict) or len(final_models) == 0:
        return None

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
    if cfg.ckpt_path is not None and str(cfg.ckpt_path).strip():
        p = Path(cfg.ckpt_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manual ckpt_path does not exist: {p}")
        return p

    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    if cfg.prefer_best_evaluated_ckpt:
        p = resolve_ckpt_from_best_eval(run_dir, preferred_dataset=cfg.preferred_eval_dataset)
        if p is not None:
            return p

    p = resolve_ckpt_from_final_index(run_dir)
    if p is not None:
        return p

    for patt in ("cnn_final__snr_*.pth", "cnn_final__noisey_*.pth", "*.pth"):
        p = resolve_newest_matching_ckpt(models_dir, patt)
        if p is not None:
            return p

    raise FileNotFoundError(f"No usable checkpoint found in {models_dir}")


def load_ckpt(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Expected checkpoint dict with key 'model_state'. Got incompatible file: {ckpt_path}")

    cfg = ckpt.get("config", {}) or {}
    state = ckpt["model_state"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    normalize_mode = str(ckpt.get("normalize_mode", "none")).lower().strip()
    normalize_stats = ckpt.get("normalize_stats", {}) or {}

    infer_cfg = {
        "use_log_psd": _coerce_bool(_get_first_present(ckpt, "use_log_psd", default=_get_first_present(cfg, "use_log_psd", default=False))),
        "log_eps": float(_get_first_present(ckpt, "log_eps", default=_get_first_present(cfg, "log_eps", default=1e-12))),
        "predict_residual_in_log": _coerce_bool(
            _get_first_present(ckpt, "predict_residual_in_log", "residual_in_log_space", default=_get_first_present(cfg, "predict_residual_in_log", "residual_in_log_space", default=False))
        ),
        "target_key": str(_get_first_present(ckpt, "target_key", default=_get_first_present(cfg, "target_key", default="Y_tilde"))),
    }
    return ckpt, cfg, state, normalize_mode, normalize_stats, infer_cfg


def infer_model_hparams_from_state(state: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "stem.0.weight" not in state:
        raise KeyError("Checkpoint missing 'stem.0.weight'; cannot infer architecture.")

    stem_w = state["stem.0.weight"]
    base_channels = int(stem_w.shape[0])
    kernel_size = int(stem_w.shape[-1])
    dropout = float(cfg.get("dropout", 0.0) or 0.0)

    arch_variant = "expanded" if any(k.startswith("body.7.") or k.startswith("body.5.") for k in state.keys()) else "legacy"
    return {
        "base_channels": base_channels,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "arch_variant": arch_variant,
    }


def build_trained_model_from_ckpt(state: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> nn.Module:
    hp = infer_model_hparams_from_state(state, cfg)
    if hp["arch_variant"] == "expanded":
        model = CNNDenoiser1DExpanded(hp["base_channels"], hp["kernel_size"], hp["dropout"])
    else:
        model = CNNDenoiser1DLegacy(hp["base_channels"], hp["kernel_size"], hp["dropout"])
    return model


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


def denormalize_linear(y_norm: np.ndarray, normalize_mode: str, normalize_stats: Optional[Dict[str, float]], ref_spec: np.ndarray, eps: float) -> np.ndarray:
    if normalize_mode == "per_sample":
        mu = float(np.mean(ref_spec))
        sig = float(np.std(ref_spec) + eps)
        return y_norm * sig + mu
    if normalize_mode == "global":
        stats = normalize_stats or {}
        mu = float(stats["mu"])
        sig = float(stats["sigma"]) + eps
        return y_norm * sig + mu
    return y_norm


@torch.no_grad()
def denoise_positive_spectrum(model: nn.Module, spec: np.ndarray, device: torch.device, normalize_mode: str, normalize_stats: Optional[Dict[str, float]], eps: float, infer_cfg: Optional[Dict[str, Any]] = None):
    infer_cfg = infer_cfg or {}
    use_log_psd = bool(infer_cfg.get("use_log_psd", False))
    log_eps = float(infer_cfg.get("log_eps", 1e-12))
    predict_residual_in_log = bool(infer_cfg.get("predict_residual_in_log", False))

    spec = np.asarray(spec, dtype=np.float64)
    spec = np.maximum(spec, 0.0)

    if use_log_psd:
        x_base = np.log10(np.maximum(spec, 0.0) + log_eps)
    else:
        x_base = spec.copy()

    if normalize_mode == "per_sample":
        x_norm, _, _ = normalize_per_sample(x_base, eps=eps)
    elif normalize_mode == "global":
        x_norm, _, _ = normalize_global(x_base, stats=(normalize_stats or {}), eps=eps)
    elif normalize_mode in ("none", "raw"):
        x_norm = x_base
    else:
        raise ValueError(f"Unsupported normalize_mode='{normalize_mode}'. Supported: per_sample, global, none.")

    xb = torch.from_numpy(x_norm.astype(np.float32))[None, None, :].to(device)
    y_model = model(xb).detach().cpu().numpy()[0, 0, :].astype(np.float64)

    if use_log_psd and predict_residual_in_log:
        y_log = x_norm - y_model
        y_log = denormalize_linear(y_log, normalize_mode, normalize_stats, x_base, eps)
        y = np.power(10.0, y_log) - log_eps
        return np.maximum(y, 0.0)

    y_lin_or_log = denormalize_linear(y_model, normalize_mode, normalize_stats, x_base, eps)

    if use_log_psd:
        y = np.power(10.0, y_lin_or_log) - log_eps
        return np.maximum(y, 0.0)

    return np.maximum(y_lin_or_log, 0.0)


def estimate_tail_floor(f: np.ndarray, y: np.ndarray, tail_frac: float) -> float:
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    n_tail = max(8, int(round(n * float(tail_frac))))
    tail = y[-min(n_tail, n):]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return float("nan")
    return float(np.median(np.maximum(tail, 0.0)))


def peak_band_mask(f: np.ndarray, y: np.ndarray, floor: float, rel_height: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    excess = np.maximum(y - floor, 0.0)
    peak = float(np.max(excess)) if excess.size else 0.0
    if peak <= 0:
        return np.ones_like(y, dtype=bool)
    mask = excess >= float(rel_height) * peak
    if not np.any(mask):
        mask[int(np.argmax(excess))] = True
    return mask


def spectrum_quality_metrics(f: np.ndarray, raw: np.ndarray, den: np.ndarray, cfg: RunConfig) -> Dict[str, float]:
    raw_floor = estimate_tail_floor(f, raw, cfg.floor_est_freq_frac)
    den_floor = estimate_tail_floor(f, den, cfg.floor_est_freq_frac)
    floor_reduction_db = 10.0 * np.log10((raw_floor + cfg.eps_std) / (den_floor + cfg.eps_std))

    raw_integral = float(np.trapz(raw, f))
    den_integral = float(np.trapz(den, f))
    raw_peak = float(np.max(raw))
    den_peak = float(np.max(den))

    band = peak_band_mask(f, raw, raw_floor, cfg.peak_band_rel_height)
    raw_excess_area = float(np.trapz(np.maximum(raw[band] - raw_floor, 0.0), f[band])) if np.count_nonzero(band) > 1 else 0.0
    den_excess_area = float(np.trapz(np.maximum(den[band] - den_floor, 0.0), f[band])) if np.count_nonzero(band) > 1 else 0.0

    return {
        "raw_floor_tail_median": float(raw_floor),
        "den_floor_tail_median": float(den_floor),
        "floor_reduction_db": float(floor_reduction_db),
        "raw_integral": raw_integral,
        "den_integral": den_integral,
        "integral_ratio": float(den_integral / max(raw_integral, cfg.eps_std)),
        "raw_peak": raw_peak,
        "den_peak": den_peak,
        "peak_ratio": float(den_peak / max(raw_peak, cfg.eps_std)),
        "raw_peakband_excess_area": raw_excess_area,
        "den_peakband_excess_area": den_excess_area,
        "peakband_excess_area_ratio": float(den_excess_area / max(raw_excess_area, cfg.eps_std)),
    }


def apply_physical_guardrails(f: np.ndarray, raw: np.ndarray, den: np.ndarray, cfg: RunConfig) -> tuple[np.ndarray, Dict[str, float]]:
    raw = np.asarray(raw, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    den = np.maximum(den, 0.0)

    if not cfg.apply_physical_guardrails:
        return den, spectrum_quality_metrics(f, raw, den, cfg)

    # Blend toward the raw spectrum only if the model violates conservation
    # bounds. This preserves successful denoising while rejecting floor collapse.
    best = den.copy()
    metrics = spectrum_quality_metrics(f, raw, best, cfg)

    lo_area = 1.0 - cfg.max_integral_change_frac
    hi_area = 1.0 + cfg.max_integral_change_frac
    lo_peak = 1.0 - cfg.max_peak_change_frac
    hi_peak = 1.0 + cfg.max_peak_change_frac

    def ok(m: Dict[str, float]) -> bool:
        return (
            lo_area <= m["integral_ratio"] <= hi_area
            and lo_peak <= m["peak_ratio"] <= hi_peak
            and m["floor_reduction_db"] >= cfg.min_floor_reduction_db
        )

    if ok(metrics):
        metrics["guardrail_blend_to_raw"] = 0.0
        return best, metrics

    chosen_alpha = 0.0
    for alpha in np.linspace(0.1, 0.9, 9):
        candidate = (1.0 - alpha) * den + alpha * raw
        cand_metrics = spectrum_quality_metrics(f, raw, candidate, cfg)
        if ok(cand_metrics):
            best = candidate
            metrics = cand_metrics
            chosen_alpha = float(alpha)
            break

    metrics["guardrail_blend_to_raw"] = chosen_alpha
    metrics["guardrail_pass"] = bool(ok(metrics))
    return best, metrics


# ==========================================
# ============== Core workflow =============
# ==========================================

def compute_auto_psd_all(channels: Dict[str, np.ndarray], fs: float, cfg: RunConfig):
    autos: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for nm, x in channels.items():
        f, Sxx = autopower_psd(
            x, fs=fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap, window=cfg.window,
            nfft=cfg.nfft, detrend=cfg.detrend,
            apply_hann_gain_corr=cfg.apply_hann_gain_corr, hann_psd_gain=cfg.hann_psd_gain,
        )
        autos[nm] = (f, Sxx)
    return autos


@torch.no_grad()
def compute_cross_and_denoise(model: nn.Module, channels: Dict[str, np.ndarray], pair_slots: List[Tuple[str, str]], fs: float, normalize_mode: str, normalize_stats: Dict[str, float], infer_cfg: Dict[str, Any], cfg: RunConfig, device: torch.device):
    autos = compute_auto_psd_all(channels, fs, cfg)
    results = []
    for a, b in pair_slots:
        fxy, Sxy = crosspower_csd(
            channels[a], channels[b], fs=fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap,
            window=cfg.window, nfft=cfg.nfft, detrend=cfg.detrend,
            apply_hann_gain_corr=cfg.apply_hann_gain_corr, hann_psd_gain=cfg.hann_psd_gain,
        )
        fa, Sxx = autos[a]
        fb, Syy = autos[b]
        if not (np.array_equal(fxy, fa) and np.array_equal(fxy, fb)):
            raise ValueError(f"Frequency axis mismatch for pair {a}, {b}")

        cross_mag = np.abs(Sxy)
        cross_phase = np.angle(Sxy)
        den_mag = denoise_positive_spectrum(
            model=model,
            spec=cross_mag,
            device=device,
            normalize_mode=normalize_mode,
            normalize_stats=normalize_stats,
            eps=cfg.eps_std,
            infer_cfg=infer_cfg,
        )
        den_mag, quality_metrics = apply_physical_guardrails(fxy, cross_mag, den_mag, cfg)
        Sxy_den = den_mag * np.exp(1j * cross_phase)
        auto_ref = np.sqrt(np.maximum(Sxx, 0.0) * np.maximum(Syy, 0.0))

        results.append({
            "a": a, "b": b, "label": pair_label(a, b), "f": fxy,
            "Sxx": Sxx, "Syy": Syy, "auto_ref": auto_ref,
            "Sxy": Sxy, "cross_mag": cross_mag, "cross_phase": cross_phase,
            "den_mag": den_mag, "Sxy_den": Sxy_den,
            "quality_metrics": quality_metrics,
        })
    return autos, results


# ==========================================
# ============== Saving helpers ============
# ==========================================

def save_pair_csvs(outdir: Path, results: List[Dict[str, Any]]):
    ensure_dir(outdir)
    for rec in results:
        df = pd.DataFrame({
            "f": rec["f"],
            "Sxx": rec["Sxx"],
            "Syy": rec["Syy"],
            "auto_ref_sqrtSxxSyy": rec["auto_ref"],
            "Re_Sxy": np.real(rec["Sxy"]),
            "Im_Sxy": np.imag(rec["Sxy"]),
            "abs_Sxy": rec["cross_mag"],
            "phase_Sxy_rad": rec["cross_phase"],
            "abs_Sxy_denoised": rec["den_mag"],
            "Re_Sxy_denoised": np.real(rec["Sxy_den"]),
            "Im_Sxy_denoised": np.imag(rec["Sxy_den"]),
        })
        df.to_csv(outdir / f"{rec['label']}.csv", index=False)


def save_auto_csvs(outdir: Path, autos: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    ensure_dir(outdir)
    for name, (f, Sxx) in autos.items():
        pd.DataFrame({"f": f, "Sxx": Sxx}).to_csv(outdir / f"{name}__Sxx.csv", index=False)


def summarize_pairs(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in results:
        metrics = dict(rec.get("quality_metrics", {}))
        row = {
            "pair": rec["label"],
            "channel_a": rec["a"],
            "channel_b": rec["b"],
            "n_freq": int(len(rec["f"])),
            "cross_mag_min": float(np.min(rec["cross_mag"])),
            "cross_mag_max": float(np.max(rec["cross_mag"])),
            "cross_mag_mean": float(np.mean(rec["cross_mag"])),
            "den_mag_min": float(np.min(rec["den_mag"])),
            "den_mag_max": float(np.max(rec["den_mag"])),
            "den_mag_mean": float(np.mean(rec["den_mag"])),
            "auto_ref_mean": float(np.mean(rec["auto_ref"])),
            "raw_integral": float(np.trapz(rec["cross_mag"], rec["f"])),
            "den_integral": float(np.trapz(rec["den_mag"], rec["f"])),
        }
        for k, v in metrics.items():
            if k not in row:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================
# ================= Main ===================
# ==========================================

def main():
    device = choose_device(CFG.device)

    synouts_root = Path(CFG.synouts_root).resolve()
    run_dir = resolve_run_dir(synouts_root, CFG.run_tag)
    ckpt_path = resolve_checkpoint(CFG, run_dir)

    outdir = run_dir / CFG.output_subdir
    ensure_dir(outdir)
    ensure_dir(outdir / "grid_figs")
    ensure_dir(outdir / "pair_figs")
    ensure_dir(outdir / "pair_csv")
    ensure_dir(outdir / "auto_csv")

    print(f"[07] Using device: {device}")
    print(f"[07] Using synouts run folder: {run_dir}")
    print(f"[07] Using checkpoint: {ckpt_path}")

    order, channels, picked_cols, t_ref, paths = load_24_channels(
        CFG.folder, CFG.pattern, CFG.time_col, CFG.value_col, CFG.expected_n,
    )

    fs = CFG.fs if CFG.fs is not None else estimate_fs_from_time(t_ref)
    print(f"[07] Using fs = {fs:.6g} Hz")

    ckpt, train_cfg, state, normalize_mode, normalize_stats, infer_cfg = load_ckpt(ckpt_path, device)
    print("[07] Loaded checkpoint metadata:")
    print(f"     best_val       = {ckpt.get('best_val', None)}")
    print(f"     best_epoch     = {ckpt.get('best_epoch', None)}")
    print(f"     normalize_mode = {normalize_mode}")
    print(f"     use_log_psd    = {infer_cfg.get('use_log_psd', False)}")
    print(f"     log_eps        = {infer_cfg.get('log_eps', 1e-12)}")
    print(f"     resid_log      = {infer_cfg.get('predict_residual_in_log', False)}")

    model = build_trained_model_from_ckpt(state, train_cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    pair_slots = build_pair_slots(order, CFG.pair_mode)

    autos, results = compute_cross_and_denoise(
        model=model,
        channels=channels,
        pair_slots=pair_slots,
        fs=fs,
        normalize_mode=normalize_mode,
        normalize_stats=normalize_stats,
        infer_cfg=infer_cfg,
        cfg=CFG,
        device=device,
    )

    auto_ref_slot = [(rec["label"], rec["f"], rec["auto_ref"]) for rec in results]
    cross_slot = [(rec["label"], rec["f"], rec["cross_mag"]) for rec in results]
    den_slot = [(rec["label"], rec["f"], rec["den_mag"]) for rec in results]

    fig_auto = plot_grid_pair_metric(auto_ref_slot, xlim_hz=CFG.xlim_hz, title_prefix=CFG.title_prefix, suptitle="Pair Reference Autopower sqrt(Sxx*Syy)", ylabel="sqrt(Sxx*Syy)", figsize=CFG.figsize_grid)
    fig_cross = plot_grid_pair_metric(cross_slot, xlim_hz=CFG.xlim_hz, title_prefix=CFG.title_prefix, suptitle="Raw Cross-power |Sxy|", ylabel="|Sxy|", figsize=CFG.figsize_grid)
    fig_den = plot_grid_pair_metric(den_slot, xlim_hz=CFG.xlim_hz, title_prefix=CFG.title_prefix, suptitle="CNN Denoised Cross-power |Sxy|", ylabel="|Sxy| denoised", figsize=CFG.figsize_grid)

    if CFG.save_figs:
        fig_auto.savefig(outdir / "grid_figs" / "pair_autoref_4x6.png", dpi=CFG.fig_dpi)
        fig_cross.savefig(outdir / "grid_figs" / "pair_cross_raw_4x6.png", dpi=CFG.fig_dpi)
        fig_den.savefig(outdir / "grid_figs" / "pair_cross_denoised_4x6.png", dpi=CFG.fig_dpi)

    for rec in results:
        fig = plot_individual_pair_comparison(rec["label"], rec["f"], rec["auto_ref"], rec["cross_mag"], rec["den_mag"], xlim_hz=CFG.xlim_hz, figsize=CFG.figsize_pair, yscale="linear", metrics=rec.get("quality_metrics"))
        if CFG.save_figs:
            fig.savefig(outdir / "pair_figs" / f"{rec['label']}__comparison.png", dpi=CFG.fig_dpi)
        plt.close(fig)

        if CFG.save_loglin_pair_figs:
            fig_log = plot_individual_pair_comparison(rec["label"], rec["f"], rec["auto_ref"], rec["cross_mag"], rec["den_mag"], xlim_hz=CFG.xlim_hz, figsize=CFG.figsize_pair, yscale="log", metrics=rec.get("quality_metrics"))
            if CFG.save_figs:
                ensure_dir(outdir / "pair_figs_loglin")
                fig_log.savefig(outdir / "pair_figs_loglin" / f"{rec['label']}__comparison_loglin.png", dpi=CFG.fig_dpi)
            plt.close(fig_log)

    if CFG.save_csv:
        save_auto_csvs(outdir / "auto_csv", autos)
        save_pair_csvs(outdir / "pair_csv", results)
        summarize_pairs(results).to_csv(outdir / "pair_summary.csv", index=False)

        meta = {
            "synouts_root": str(synouts_root),
            "run_dir": str(run_dir),
            "ckpt_path": str(ckpt_path),
            "normalize_mode": normalize_mode,
            "normalize_stats": normalize_stats,
            "inference_config": infer_cfg,
            "picked_value_columns": picked_cols,
            "cece_files": [str(p.resolve()) for p in paths],
            "pair_mode": CFG.pair_mode,
            "pair_slots": [{"a": a, "b": b, "label": pair_label(a, b)} for a, b in pair_slots],
            "spectral_params": {
                "fs": float(fs),
                "nperseg": int(CFG.nperseg),
                "noverlap": int(CFG.noverlap),
                "window": str(CFG.window),
                "nfft": None if CFG.nfft is None else int(CFG.nfft),
                "detrend": str(CFG.detrend),
                "apply_hann_gain_corr": bool(CFG.apply_hann_gain_corr),
                "hann_psd_gain": float(CFG.hann_psd_gain),
            },
            "physical_guardrails": {
                "enabled": bool(CFG.apply_physical_guardrails),
                "floor_est_freq_frac": float(CFG.floor_est_freq_frac),
                "peak_band_rel_height": float(CFG.peak_band_rel_height),
                "min_floor_reduction_db": float(CFG.min_floor_reduction_db),
                "max_integral_change_frac": float(CFG.max_integral_change_frac),
                "max_peak_change_frac": float(CFG.max_peak_change_frac),
            },
            "train_config": train_cfg,
            "note": "CNN denoises |Sxy| only; original phase is reattached to form Sxy_denoised.",
        }
        with open(outdir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"[07] Outputs saved under: {outdir}")
    plt.show()


if __name__ == "__main__":
    main()
