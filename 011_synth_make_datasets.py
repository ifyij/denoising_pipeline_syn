# 01_synth_make_datasets.py
"""
Synthetic dataset generator for PSD denoising:
    Y_obs(f) [V^2/Hz]  ->  Y_tilde(f) [V^2/Hz]

This version is explicitly formulated for PSD denoising, not time-series denoising.

What is generated
-----------------
For each sample:
- Y_tilde : exact clean PSD target actually used
            (signal + smooth baseline floor)
- Y_floor : exact additive noisy residual actually used
            so that Y_obs = Y_tilde + Y_floor
- Y_obs   : noisy observed PSD made directly in PSD/frequency space
- X       : optional synthetic time series for visualization only

Important note:
- X is NOT used for training in the PSD-denoising pipeline.
- Y_obs is generated directly in PSD space, so X is only a plausible visualization artifact
  and is not required to be the exact source of Y_obs.
- By construction:
      Y_obs = Y_tilde + Y_floor
  up to float precision.

Run-folder naming convention
----------------------------
Each run writes to:
    C:/synruns/<RUN_ID>/
        config.json
        manifest.json
        datasets/
        plots/

Dataset filenames inside a run are stable:
- train_gen1_clean.npz
- train_gen2_snrXXdB.npz
- train_gen3_mix_50_50.npz
- eval_mixed_snr.npz
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Configuration (edit here)
# -----------------------------

@dataclass
class Config:
    # Repro
    seed: int = 123

    # Short Windows-safe root folder containing all runs
    runs_root: str = "C:/synruns"

    # Optional: if set (non-empty), use this exact run_id; otherwise auto-generate
    run_id: str = ""

    # Frequency / optional time-series settings
    fs: float = 1e6
    T: int = 65536

    # Frequency grid used for PSD arrays
    # Keep the same F = nperseg//2 + 1 convention as Welch for compatibility
    nperseg: int = 4096

    # Gaussian PSD target parameter ranges
    f0_min: float = 5e3
    f0_max: float = 120e3
    deltaf_min: float = 1e3
    deltaf_max: float = 30e3

    # Peak amplitude range of signal-only Gaussian PSD target
    A_min: float = 5e-10
    A_max: float = 2e-8

    # Dataset sizes
    N_gen1: int = 8000
    N_gen2_per_snr: int = 4000
    N_gen3: int = 8000
    N_eval: int = 2000

    # These are interpreted as PEAK-TO-SMOOTH-FLOOR ratios in PSD space:
    #   peak_snr_db = 10*log10( peak(signal_only) / nominal_floor )
    GEN2_SNR_DB_LIST: Tuple[float, ...] = (0.0, -3.0)
    GEN3_SNR_DB_LIST: Tuple[float, ...] = (3.0, 0.0, -3.0, -6.0, -8.0)
    EVAL_SNR_DB_LIST: Tuple[float, ...] = (6.0, 3.0, 0.0, -3.0, -6.0, -8.0, -10.0)
    # Residual hard minimum floor so PSD never hits zero
    min_noise_floor_v2hz: float = 1.0e-12

    # Smooth baseline floor model
    floor_tilt_frac: float = 0.04
    floor_wobble_frac: float = 0.03

    # Jagged residual around the smooth baseline
    floor_jitter_frac: float = 0.08
    floor_corr_bins: int = 3

    # Optional weak low-frequency broadband excess in the SMOOTH baseline
    # Set lf_excess_frac = 0.0 if you want a very flat baseline only.
    lf_excess_frac: float = 0.20
    lf_excess_fc_hz: float = 3.0e4
    lf_excess_power: float = 1.2

    # Sanity plots
    n_plot_examples: int = 3
    max_time_plot_samples: int = 4000

    # Optional plotting x-limit for full-band PSD figures
    plot_max_freq_hz: float = 150e3

    floor_gain: float = 2.0

    eps: float = 1e-30


# -----------------------------
# Naming + manifest helpers
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _snr_piece(x: float) -> str:
    if x < 0:
        return f"m{abs(x):g}".replace(".", "p")
    return f"{x:g}".replace(".", "p")


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_manifest(run_dir: Path, cfg: Config, dataset_paths: Dict[str, Path]) -> None:
    files = []
    for key, p in dataset_paths.items():
        if p.exists():
            files.append(
                {
                    "label": key,
                    "path": str(p.relative_to(run_dir)),
                    "bytes": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
            )

    manifest = {
        "run_id": run_dir.name,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
        "files": files,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def make_run_id(cfg: Config) -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    core = {
        "noise_model": "clean_target_includes_smooth_floor_obs_adds_residual",
        "GEN2_SNR_DB_LIST": cfg.GEN2_SNR_DB_LIST,
        "GEN3_SNR_DB_LIST": cfg.GEN3_SNR_DB_LIST,
        "EVAL_SNR_DB_LIST": cfg.EVAL_SNR_DB_LIST,
        "min_noise_floor_v2hz": cfg.min_noise_floor_v2hz,
        "floor_tilt_frac": cfg.floor_tilt_frac,
        "floor_wobble_frac": cfg.floor_wobble_frac,
        "floor_jitter_frac": cfg.floor_jitter_frac,
        "lf_excess_frac": cfg.lf_excess_frac,
        "lf_excess_fc_hz": cfg.lf_excess_fc_hz,
        "lf_excess_power": cfg.lf_excess_power,
        "floor_corr_bins": cfg.floor_corr_bins,
        "seed": cfg.seed,
        "fs": cfg.fs,
        "T": cfg.T,
        "nperseg": cfg.nperseg,
        "f0": (cfg.f0_min, cfg.f0_max),
        "deltaf": (cfg.deltaf_min, cfg.deltaf_max),
        "A": (cfg.A_min, cfg.A_max),
    }
    core_json = json.dumps(core, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(core_json).hexdigest()[:8]
    return f"synv7_{ts}_{h}"


# -----------------------------
# Core math helpers
# -----------------------------

def sample_log_uniform(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    if lo <= 0 or hi <= 0:
        raise ValueError("Log-uniform requires positive bounds.")
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def gaussian_psd(f: np.ndarray, f0: float, deltaf: float, A: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((f - f0) / deltaf) ** 2)


def make_time_series_from_psd(
    rng: np.random.Generator,
    fs: float,
    T: int,
    target_psd_rfft: np.ndarray,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Optional visualization-only time series from a target one-sided PSD on the rFFT grid.
    """
    z = rng.standard_normal(target_psd_rfft.size) + 1j * rng.standard_normal(target_psd_rfft.size)
    shape = np.sqrt(np.maximum(target_psd_rfft, 0.0) + eps)
    Xf = z * shape

    Xf[0] = np.real(Xf[0]) + 0j
    if T % 2 == 0:
        Xf[-1] = np.real(Xf[-1]) + 0j

    x = np.fft.irfft(Xf, n=T)

    f_rfft = np.fft.rfftfreq(T, d=1.0 / fs)
    df = f_rfft[1] - f_rfft[0] if f_rfft.size > 1 else fs

    var_target = float(np.sum(target_psd_rfft) * df)
    var_current = float(np.mean(x**2))
    if var_current > 0 and var_target > 0:
        x *= np.sqrt(var_target / var_current)

    return x.astype(np.float32)


def make_freq_grid(cfg: Config) -> np.ndarray:
    """
    PSD frequency grid. Matches Welch one-sided convention length:
      F = nperseg // 2 + 1
    """
    return np.fft.rfftfreq(cfg.nperseg, d=1.0 / cfg.fs).astype(np.float32)


def floor_level_from_peak_snr(peak_value: float, peak_snr_db: float, eps: float = 1e-30) -> float:
    """
    peak_snr_db = 10*log10(peak_value / nominal_floor)
    """
    snr_lin = 10.0 ** (peak_snr_db / 10.0)
    return peak_value / max(snr_lin, eps)


# -----------------------------
# PSD-space clean baseline + residual noise model
# -----------------------------

def make_smooth_floor_profile(
    rng: np.random.Generator,
    f: np.ndarray,
    nominal_floor: float,
    min_noise_floor: float,
    floor_tilt_frac: float,
    floor_wobble_frac: float,
    lf_excess_frac: float,
    lf_excess_fc_hz: float,
    lf_excess_power: float,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Smooth baseline floor only (no jagged noise).
    """
    f = np.asarray(f, dtype=np.float64)
    u = (f - f.min()) / max(f.max() - f.min(), eps)

    tilt = 1.0 + floor_tilt_frac * (u - 0.5)

    phase1 = rng.uniform(0.0, 2.0 * np.pi)
    phase2 = rng.uniform(0.0, 2.0 * np.pi)
    wobble = (
        1.0
        + floor_wobble_frac * np.sin(2.0 * np.pi * 1.1 * u + phase1)
        + 0.5 * floor_wobble_frac * np.sin(2.0 * np.pi * 2.4 * u + phase2)
    )

    if lf_excess_frac > 0.0:
        lf_shape = 1.0 / np.power(
            1.0 + np.maximum(f, 0.0) / max(lf_excess_fc_hz, eps),
            lf_excess_power
        )
        lf_shape = lf_shape / max(float(np.max(lf_shape)), eps)
        lf_term = 1.0 + lf_excess_frac * lf_shape
    else:
        lf_term = np.ones_like(f)

    smooth_floor = nominal_floor * tilt * wobble * lf_term
    smooth_floor = np.maximum(smooth_floor, min_noise_floor)

    return smooth_floor.astype(np.float32)


def make_jitter_profile(
    rng: np.random.Generator,
    smooth_floor: np.ndarray,
    floor_jitter_frac: float,
    floor_corr_bins: int,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Jagged random component around zero.
    """
    smooth_floor = np.asarray(smooth_floor, dtype=np.float64)

    sigma = np.maximum(floor_jitter_frac * smooth_floor, eps)
    jitter = rng.normal(0.0, sigma, size=smooth_floor.size)

    if floor_corr_bins > 1:
        k = int(floor_corr_bins)
        kernel = np.ones(k, dtype=np.float64) / float(k)
        jitter = np.convolve(jitter, kernel, mode="same")

    return jitter.astype(np.float32)


def make_noisy_psd_from_clean(
    rng: np.random.Generator,
    f: np.ndarray,
    y_clean_signal: np.ndarray,
    peak_snr_db: float,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    New definition:

        Y_tilde = signal + smooth_floor
        Y_obs   = Y_tilde + jitter

    Returns
    -------
    y_obs : observed PSD
    y_tilde_used : clean target actually used
    y_jitter_used : additive noisy residual actually used, so that:
                    y_obs = y_tilde_used + y_jitter_used
    """
    y_clean_signal = np.asarray(y_clean_signal, dtype=np.float32).copy()

    peak_value = float(np.max(y_clean_signal))
    nominal_floor = floor_level_from_peak_snr(peak_value, peak_snr_db, eps=cfg.eps)
    nominal_floor = max(nominal_floor, cfg.min_noise_floor_v2hz)
    nominal_floor *= cfg.floor_gain

    y_smooth_floor = make_smooth_floor_profile(
        rng=rng,
        f=f,
        nominal_floor=nominal_floor,
        min_noise_floor=cfg.min_noise_floor_v2hz,
        floor_tilt_frac=cfg.floor_tilt_frac,
        floor_wobble_frac=cfg.floor_wobble_frac,
        lf_excess_frac=cfg.lf_excess_frac,
        lf_excess_fc_hz=cfg.lf_excess_fc_hz,
        lf_excess_power=cfg.lf_excess_power,
        eps=cfg.eps,
    ).astype(np.float32)

    y_tilde_used = y_clean_signal + y_smooth_floor

    y_jitter = make_jitter_profile(
        rng=rng,
        smooth_floor=y_smooth_floor,
        floor_jitter_frac=cfg.floor_jitter_frac,
        floor_corr_bins=cfg.floor_corr_bins,
        eps=cfg.eps,
    ).astype(np.float32)

    y_obs = y_tilde_used + y_jitter
    y_obs = np.maximum(y_obs, cfg.min_noise_floor_v2hz).astype(np.float32)

    return y_obs, y_tilde_used.astype(np.float32), y_jitter.astype(np.float32)


# -----------------------------
# Plotting / saving
# -----------------------------

def plot_examples(
    out_dir: Path,
    tag: str,
    fs: float,
    X: np.ndarray,
    f: np.ndarray,
    Y_tilde: np.ndarray,
    Y_obs: np.ndarray,
    Y_floor: np.ndarray,
    params: np.ndarray,
    max_time_plot_samples: int,
    n_examples: int,
    rng: np.random.Generator,
    plot_max_freq_hz: float | None = None,
) -> None:
    ensure_dir(out_dir)

    N = X.shape[0]
    idx = rng.choice(N, size=min(n_examples, N), replace=False)

    # Time series
    plt.figure(figsize=(12, 7))
    for k, i in enumerate(idx, start=1):
        t = np.arange(min(max_time_plot_samples, X.shape[1])) / fs
        snr_val = params[i, 3]
        snr_str = "clean" if int(params[i, 4]) == 0 else f"{snr_val:.1f} dB"
        plt.plot(t, X[i, :t.size], label=f"ex {i} (noisy={int(params[i,4])}, peak_snr={snr_str})")
        if k >= 6:
            break

    plt.xlabel("t (s)")
    plt.ylabel("Amplitude (arb. / V-like)")
    plt.title(f"{tag}: example synthetic time series (visualization only)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_time_examples.png", dpi=200)
    plt.close()

    # One PSD figure per example
    for i in idx:
        y_obs_i = np.asarray(Y_obs[i], dtype=np.float64)
        y_til_i = np.asarray(Y_tilde[i], dtype=np.float64)
        y_floor_i = np.asarray(Y_floor[i], dtype=np.float64)

        f0_i = float(params[i, 0])
        df_i = float(params[i, 1])
        amp_i = float(params[i, 2])
        snr_val = params[i, 3]
        is_noisy = int(params[i, 4])
        snr_str = "clean" if is_noisy == 0 else f"{snr_val:.1f} dB"

        plt.figure(figsize=(11, 6))
        plt.plot(f, y_obs_i, label="Y_obs", linewidth=1.6)
        plt.plot(f, y_til_i, "--", label="Y_tilde (true used)", linewidth=1.8)
        plt.plot(f, y_floor_i, ":", label="noise residual (true used)", linewidth=1.8)

        if plot_max_freq_hz is not None and plot_max_freq_hz > 0:
            plt.xlim(0.0, min(float(f[-1]), float(plot_max_freq_hz)))

        plt.xlabel("f (Hz)")
        plt.ylabel(r"PSD (V$^2$/Hz)")
        plt.title(
            f"{tag}: ex {i} | noisy={is_noisy} | peak_snr={snr_str}\n"
            f"f0={f0_i:.1f} Hz, deltaf={df_i:.1f} Hz, A={amp_i:.3e}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_psd_ex_{i}.png", dpi=220)
        plt.close()

        f_lo = max(0.0, f0_i - 4.0 * df_i)
        f_hi = min(float(f[-1]), f0_i + 4.0 * df_i)
        mask = (f >= f_lo) & (f <= f_hi)

        if np.any(mask):
            plt.figure(figsize=(11, 6))
            plt.plot(f[mask], y_obs_i[mask], label="Y_obs", linewidth=1.8)
            plt.plot(f[mask], y_til_i[mask], "--", label="Y_tilde (true used)", linewidth=2.0)
            plt.plot(f[mask], y_floor_i[mask], ":", label="noise residual (true used)", linewidth=1.8)

            plt.xlabel("f (Hz)")
            plt.ylabel(r"PSD (V$^2$/Hz)")
            plt.title(
                f"{tag}: ex {i} (zoom near peak)\n"
                f"f0={f0_i:.1f} Hz, deltaf={df_i:.1f} Hz, A={amp_i:.3e}, peak_snr={snr_str}"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{tag}_psd_ex_{i}_zoom.png", dpi=220)
            plt.close()


def save_dataset_npz(
    out_path: Path,
    X: np.ndarray,
    f: np.ndarray,
    Y_tilde: np.ndarray,
    Y_obs: np.ndarray,
    Y_floor: np.ndarray,
    params: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        f=f.astype(np.float32),
        Y_tilde=Y_tilde.astype(np.float32),
        Y_obs=Y_obs.astype(np.float32),
        Y_floor=Y_floor.astype(np.float32),
        params=params.astype(np.float32),
        meta_json=json.dumps(meta),
    )


# -----------------------------
# Dataset builder
# -----------------------------

def build_dataset(
    cfg: Config,
    rng: np.random.Generator,
    N: int,
    noisy: bool,
    snr_db_choices: Tuple[float, ...] | None,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create dataset arrays:
      X: (N, T) optional synthetic time series for visualization
      f: (F,) frequency grid
      Y_tilde: (N, F) exact clean target used (signal + smooth floor)
      Y_obs: (N, F) noisy observed PSDs generated in PSD space
      Y_floor: (N, F) exact additive noisy residual used
      params: (N, 5) = [f0, deltaf, A, peak_snr_db, is_noisy]
    """
    fs, T = cfg.fs, cfg.T
    f = make_freq_grid(cfg)
    F = f.size

    # Separate rFFT grid for optional time-series visualization
    f_r = np.fft.rfftfreq(T, d=1.0 / fs).astype(np.float32)

    # Sample clean Gaussian parameters
    f0 = rng.uniform(cfg.f0_min, cfg.f0_max, size=N).astype(np.float32)
    deltaf = rng.uniform(cfg.deltaf_min, cfg.deltaf_max, size=N).astype(np.float32)
    A = sample_log_uniform(rng, cfg.A_min, cfg.A_max, size=N).astype(np.float32)

    meta_extra: Dict[str, Any] = {}

    if noisy:
        if not snr_db_choices or len(snr_db_choices) == 0:
            raise ValueError("Noisy dataset requested but snr_db_choices is empty.")
        snr_db_choices = tuple(float(v) for v in snr_db_choices)
        peak_snr_db = rng.choice(np.array(snr_db_choices, dtype=np.float32), size=N, replace=True)
        is_noisy = np.ones(N, dtype=np.float32)

        meta_extra.update(
            {
                "noise_model": "clean_target_includes_smooth_floor_obs_adds_residual",
                "snr_db_choices": [float(v) for v in snr_db_choices],
                "min_noise_floor_v2hz": float(cfg.min_noise_floor_v2hz),
                "floor_tilt_frac": float(cfg.floor_tilt_frac),
                "floor_wobble_frac": float(cfg.floor_wobble_frac),
                "floor_jitter_frac": float(cfg.floor_jitter_frac),
                "lf_excess_frac": float(cfg.lf_excess_frac),
                "lf_excess_fc_hz": float(cfg.lf_excess_fc_hz),
                "lf_excess_power": float(cfg.lf_excess_power),
                "floor_corr_bins": int(cfg.floor_corr_bins),
            }
        )
    else:
        peak_snr_db = np.full(N, np.inf, dtype=np.float32)
        is_noisy = np.zeros(N, dtype=np.float32)

        meta_extra.update(
            {
                "noise_model": "clean_target_with_tiny_floor_no_residual",
                "min_noise_floor_v2hz": float(cfg.min_noise_floor_v2hz),
            }
        )

    X = np.zeros((N, T), dtype=np.float32)
    Y_tilde = np.zeros((N, F), dtype=np.float32)
    Y_obs = np.zeros((N, F), dtype=np.float32)
    Y_floor = np.zeros((N, F), dtype=np.float32)

    for i in range(N):
        y_clean_signal = gaussian_psd(
            f, float(f0[i]), float(deltaf[i]), float(A[i])
        ).astype(np.float32)

        y_clean_r = gaussian_psd(
            f_r, float(f0[i]), float(deltaf[i]), float(A[i])
        ).astype(np.float32)
        x_viz = make_time_series_from_psd(rng, fs, T, y_clean_r, eps=cfg.eps)

        if noisy:
            y_obs_i, y_tilde_used_i, y_resid_i = make_noisy_psd_from_clean(
                rng=rng,
                f=f,
                y_clean_signal=y_clean_signal,
                peak_snr_db=float(peak_snr_db[i]),
                cfg=cfg,
            )
        else:
            y_tilde_used_i = np.maximum(
                y_clean_signal + cfg.min_noise_floor_v2hz,
                cfg.min_noise_floor_v2hz
            ).astype(np.float32)
            y_resid_i = np.zeros_like(y_tilde_used_i, dtype=np.float32)
            y_obs_i = y_tilde_used_i.copy()

        X[i] = x_viz
        Y_tilde[i] = y_tilde_used_i
        Y_floor[i] = y_resid_i
        Y_obs[i] = y_obs_i

        if (i + 1) % 500 == 0:
            print(f"[{split_name}] generated {i+1}/{N}")

    params = np.column_stack([f0, deltaf, A, peak_snr_db.astype(np.float32), is_noisy]).astype(np.float32)
    return X, f, Y_tilde, Y_obs, Y_floor, params, meta_extra


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    run_id = cfg.run_id.strip() if cfg.run_id.strip() else make_run_id(cfg)

    runs_root = Path(cfg.runs_root)
    run_dir = runs_root / run_id
    ensure_dir(run_dir)

    datasets_dir = run_dir / "datasets"
    plots_dir = run_dir / "plots"
    ensure_dir(datasets_dir)
    ensure_dir(plots_dir)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f_out:
        json.dump(asdict(cfg), f_out, indent=2)

    written: Dict[str, Path] = {}

    # Gen 1: clean-only
    print("\n=== Building train_gen1 (clean-only) ===")
    X1, f, Yt1, Yo1, Yf1, p1, meta1 = build_dataset(
        cfg=cfg,
        rng=rng,
        N=cfg.N_gen1,
        noisy=False,
        snr_db_choices=None,
        split_name="train_gen1",
    )
    p_gen1 = datasets_dir / "train_gen1_clean.npz"
    save_dataset_npz(
        p_gen1,
        X1, f, Yt1, Yo1, Yf1, p1,
        meta={"split": "train_gen1", "noisy": False, **meta1},
    )
    plot_examples(
        plots_dir, "train_gen1_clean", cfg.fs, X1, f, Yt1, Yo1, Yf1, p1,
        cfg.max_time_plot_samples, cfg.n_plot_examples, rng, cfg.plot_max_freq_hz
    )
    written["train_gen1_clean"] = p_gen1

    # Gen 2: one file per SNR bucket
    for snr_db in cfg.GEN2_SNR_DB_LIST:
        print(f"\n=== Building train_gen2 (noisy) @ peak_snr_db={snr_db:.1f} dB ===")
        X2, f2, Yt2, Yo2, Yf2, p2, meta2 = build_dataset(
            cfg=cfg,
            rng=rng,
            N=cfg.N_gen2_per_snr,
            noisy=True,
            snr_db_choices=(float(snr_db),),
            split_name=f"train_gen2_snr{snr_db:.1f}dB",
        )
        assert np.allclose(f2, f)

        fname = f"train_gen2_snr{_snr_piece(float(snr_db))}dB.npz"
        p_gen2 = datasets_dir / fname
        save_dataset_npz(
            p_gen2,
            X2, f, Yt2, Yo2, Yf2, p2,
            meta={"split": "train_gen2", "noisy": True, "peak_snr_db_bucket": float(snr_db), **meta2},
        )
        plot_examples(
            plots_dir, f"train_gen2_snr{_snr_piece(float(snr_db))}dB",
            cfg.fs, X2, f, Yt2, Yo2, Yf2, p2,
            cfg.max_time_plot_samples, cfg.n_plot_examples, rng, cfg.plot_max_freq_hz
        )
        written[f"train_gen2_snr{snr_db:.1f}dB"] = p_gen2

    # Gen 3: 50% clean / 50% noisy
    print("\n=== Building train_gen3_mix (50% clean / 50% noisy) ===")
    N3 = cfg.N_gen3
    N3_clean = N3 // 2
    N3_noisy = N3 - N3_clean

    X3c, f3, Yt3c, Yo3c, Yf3c, p3c, _meta3c = build_dataset(
        cfg=cfg,
        rng=rng,
        N=N3_clean,
        noisy=False,
        snr_db_choices=None,
        split_name="train_gen3_clean_half",
    )
    X3n, f3b, Yt3n, Yo3n, Yf3n, p3n, meta3n = build_dataset(
        cfg=cfg,
        rng=rng,
        N=N3_noisy,
        noisy=True,
        snr_db_choices=cfg.GEN3_SNR_DB_LIST,
        split_name="train_gen3_noisy_half",
    )
    assert np.allclose(f3, f3b)

    X3 = np.concatenate([X3c, X3n], axis=0)
    Yt3 = np.concatenate([Yt3c, Yt3n], axis=0)
    Yo3 = np.concatenate([Yo3c, Yo3n], axis=0)
    Yf3 = np.concatenate([Yf3c, Yf3n], axis=0)
    p3 = np.concatenate([p3c, p3n], axis=0)

    perm = rng.permutation(N3)
    X3, Yt3, Yo3, Yf3, p3 = X3[perm], Yt3[perm], Yo3[perm], Yf3[perm], p3[perm]

    p_gen3 = datasets_dir / "train_gen3_mix_50_50.npz"
    save_dataset_npz(
        p_gen3,
        X3, f3, Yt3, Yo3, Yf3, p3,
        meta={
            "split": "train_gen3",
            "noisy": "mixed_50_50",
            "gen3_snr_db_choices": list(cfg.GEN3_SNR_DB_LIST),
            **meta3n,
        },
    )
    plot_examples(
        plots_dir, "train_gen3_mix_50_50", cfg.fs, X3, f3, Yt3, Yo3, Yf3, p3,
        cfg.max_time_plot_samples, cfg.n_plot_examples, rng, cfg.plot_max_freq_hz
    )
    written["train_gen3_mix_50_50"] = p_gen3

    # Eval
    print("\n=== Building eval (held-out; noisy uses SNR buckets) ===")
    Xe, fe, Yte, Yoe, Yfe, pe, metae = build_dataset(
        cfg=cfg,
        rng=rng,
        N=cfg.N_eval,
        noisy=True,
        snr_db_choices=cfg.EVAL_SNR_DB_LIST,
        split_name="eval",
    )
    p_eval = datasets_dir / "eval_mixed_snr.npz"
    save_dataset_npz(
        p_eval,
        Xe, fe, Yte, Yoe, Yfe, pe,
        meta={"split": "eval", "snr_db_choices": list(cfg.EVAL_SNR_DB_LIST), **metae},
    )
    plot_examples(
        plots_dir, "eval_mixed_snr", cfg.fs, Xe, fe, Yte, Yoe, Yfe, pe,
        cfg.max_time_plot_samples, cfg.n_plot_examples, rng, cfg.plot_max_freq_hz
    )
    written["eval_mixed_snr"] = p_eval

    # Manifest
    write_manifest(run_dir, cfg, written)

    print("\nDone.")
    print(f"RUN_ID:          {run_id}")
    print(f"Run folder:      {run_dir.resolve()}")
    print(f"Datasets folder: {datasets_dir.resolve()}")
    print(f"Plots folder:    {plots_dir.resolve()}")
    print(f"Manifest:        {(run_dir / 'manifest.json').resolve()}")

    # Exact consistency check on one sample from eval
    i = 0
    err = np.max(np.abs(Yoe[i] - (Yte[i] + Yfe[i])))
    print(f"\nSanity check max|Y_obs - (Y_tilde + Y_floor)| on eval[0]: {err:.3e}")


if __name__ == "__main__":
    main()