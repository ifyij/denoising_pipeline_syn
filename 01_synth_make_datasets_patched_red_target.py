# 01_synth_make_datasets_patched_time_welch.py
"""
Synthetic dataset generator for PSD denoising:
    Y_obs(f) [V^2/Hz]  ->  Y_target(f) [V^2/Hz]

PATCHED TO ADDRESS:
1) PSD-domain train/test mismatch:
   - We now generate time-domain signals/noise and compute PSDs with the SAME
     Welch/CSD-style estimator used later in the CECE pipeline.

2) Overly aggressive target:
   - The target floor is reduced less aggressively than before.

3) "Too easy" vertical-shift task:
   - Because both Y_obs and Y_target come from finite-length time series + Welch,
     both contain estimator roughness, so the mapping is no longer just a smooth
     constant downward offset in log space.

4) Stable naming / folder conventions:
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
from typing import Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


@dataclass
class Config:
    seed: int = 123
    runs_root: str = "C:/synruns"
    run_id: str = ""

    # Time / Welch config
    fs: float = 1e6
    T: int = 65536
    nperseg: int = 4096
    noverlap: int = 2048
    window: str = "hann"
    detrend: str = "constant"
    scaling: str = "density"

    # Gaussian peak parameter ranges
    f0_min: float = 5e3
    f0_max: float = 120e3
    deltaf_min: float = 1e3
    deltaf_max: float = 30e3
    A_min: float = 5e-10
    A_max: float = 2e-8

    # Dataset sizes
    N_gen1: int = 8000
    N_gen2_per_snr: int = 4000
    N_gen3: int = 8000
    N_eval: int = 2000

    # Peak signal-to-observation-floor ratio buckets
    GEN2_SNR_DB_LIST: Tuple[float, ...] = (3.0, 0.0, -3.0, -6.0)
    GEN3_SNR_DB_LIST: Tuple[float, ...] = (6.0, 3.0, 0.0, -3.0, -6.0, -8.0)
    EVAL_SNR_DB_LIST: Tuple[float, ...] = (8.0, 6.0, 3.0, 0.0, -3.0, -6.0, -8.0, -10.0)

    # Floor controls
    min_noise_floor_v2hz: float = 1.0e-13
    obs_floor_gain: float = 2.2

    floor_tilt_frac: float = 0.10
    floor_wobble_frac: float = 0.08
    lf_excess_frac: float = 0.90
    lf_excess_fc_hz: float = 2.5e4
    lf_excess_power: float = 1.35

    # Make target less aggressive than before
    target_floor_frac_of_obs: float = 0.50
    target_floor_tilt_frac: float = 0.04
    target_floor_wobble_frac: float = 0.03
    target_hf_rolloff_fc_hz: float = 1.1e5
    target_hf_rolloff_power: float = 1.8

    # Shoulder / secondary structure
    shoulder_prob: float = 0.45
    shoulder_amp_frac_min: float = 0.04
    shoulder_amp_frac_max: float = 0.18
    shoulder_offset_hz_min: float = 8e3
    shoulder_offset_hz_max: float = 40e3
    shoulder_width_frac_min: float = 1.3
    shoulder_width_frac_max: float = 2.6

    # Time-domain roughness / modulation
    amplitude_jitter_frac: float = 0.10
    obs_noise_extra_gain_min: float = 0.95
    obs_noise_extra_gain_max: float = 1.10
    tgt_noise_extra_gain_min: float = 0.95
    tgt_noise_extra_gain_max: float = 1.08

    n_plot_examples: int = 3
    max_time_plot_samples: int = 4000
    plot_max_freq_hz: float = 150e3
    eps: float = 1e-30


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
        "noise_model": "time_domain_then_welch",
        "GEN2_SNR_DB_LIST": cfg.GEN2_SNR_DB_LIST,
        "GEN3_SNR_DB_LIST": cfg.GEN3_SNR_DB_LIST,
        "EVAL_SNR_DB_LIST": cfg.EVAL_SNR_DB_LIST,
        "target_floor_frac_of_obs": cfg.target_floor_frac_of_obs,
        "obs_floor_gain": cfg.obs_floor_gain,
        "seed": cfg.seed,
        "fs": cfg.fs,
        "T": cfg.T,
        "nperseg": cfg.nperseg,
        "noverlap": cfg.noverlap,
    }
    h = hashlib.sha1(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"synwelch_{ts}_{h}"


def sample_log_uniform(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    if lo <= 0 or hi <= 0:
        raise ValueError("Log-uniform requires positive bounds.")
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def gaussian_psd(f: np.ndarray, f0: float, deltaf: float, A: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((f - f0) / deltaf) ** 2)


def maybe_make_shoulder(
    rng: np.random.Generator,
    f: np.ndarray,
    f0: float,
    deltaf: float,
    A: float,
    cfg: Config,
) -> np.ndarray:
    if rng.uniform() > cfg.shoulder_prob:
        return np.zeros_like(f, dtype=np.float32)

    sign = -1.0 if rng.uniform() < 0.35 else 1.0
    offset = sign * rng.uniform(cfg.shoulder_offset_hz_min, cfg.shoulder_offset_hz_max)
    f1 = float(np.clip(f0 + offset, f[0], f[-1]))
    width = float(deltaf * rng.uniform(cfg.shoulder_width_frac_min, cfg.shoulder_width_frac_max))
    amp = float(A * rng.uniform(cfg.shoulder_amp_frac_min, cfg.shoulder_amp_frac_max))
    return gaussian_psd(f, f1, width, amp).astype(np.float32)


def floor_level_from_peak_snr(peak_value: float, peak_snr_db: float, eps: float = 1e-30) -> float:
    snr_lin = 10.0 ** (peak_snr_db / 10.0)
    return peak_value / max(snr_lin, eps)


def make_smooth_floor_profile(
    rng: np.random.Generator,
    f: np.ndarray,
    nominal_floor: float,
    min_noise_floor: float,
    tilt_frac: float,
    wobble_frac: float,
    lf_excess_frac: float,
    lf_excess_fc_hz: float,
    lf_excess_power: float,
    eps: float = 1e-30,
) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    u = (f - f.min()) / max(f.max() - f.min(), eps)

    tilt = 1.0 + tilt_frac * (u - 0.5)

    phase1 = rng.uniform(0.0, 2.0 * np.pi)
    phase2 = rng.uniform(0.0, 2.0 * np.pi)
    wobble = (
        1.0
        + wobble_frac * np.sin(2.0 * np.pi * 1.05 * u + phase1)
        + 0.6 * wobble_frac * np.sin(2.0 * np.pi * 2.4 * u + phase2)
    )

    lf_shape = 1.0 / np.power(
        1.0 + np.maximum(f, 0.0) / max(lf_excess_fc_hz, eps),
        lf_excess_power
    )
    lf_shape /= max(float(np.max(lf_shape)), eps)
    lf_term = 1.0 + lf_excess_frac * lf_shape

    y = nominal_floor * tilt * wobble * lf_term
    return np.maximum(y, min_noise_floor).astype(np.float32)


def make_target_floor(
    rng: np.random.Generator,
    f: np.ndarray,
    obs_floor: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    obs_floor = np.asarray(obs_floor, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    u = (f - f.min()) / max(f.max() - f.min(), cfg.eps)

    base = np.maximum(obs_floor * cfg.target_floor_frac_of_obs, cfg.min_noise_floor_v2hz)

    tilt = 1.0 + cfg.target_floor_tilt_frac * (u - 0.5)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    wobble = 1.0 + cfg.target_floor_wobble_frac * np.sin(2.0 * np.pi * 1.3 * u + phase)

    rolloff = 1.0 / np.power(
        1.0 + np.maximum(f, 0.0) / max(cfg.target_hf_rolloff_fc_hz, cfg.eps),
        cfg.target_hf_rolloff_power
    )
    rolloff /= max(float(np.max(rolloff)), cfg.eps)
    rolloff = 0.55 + 0.45 * rolloff

    y = base * tilt * wobble * rolloff
    return np.maximum(y, cfg.min_noise_floor_v2hz).astype(np.float32)


def make_time_series_from_one_sided_psd(
    rng: np.random.Generator,
    fs: float,
    T: int,
    psd_rfft: np.ndarray,
    eps: float = 1e-30,
) -> np.ndarray:
    z = rng.standard_normal(psd_rfft.size) + 1j * rng.standard_normal(psd_rfft.size)
    Xf = z * np.sqrt(np.maximum(psd_rfft, 0.0) + eps)
    Xf[0] = np.real(Xf[0]) + 0j
    if T % 2 == 0:
        Xf[-1] = np.real(Xf[-1]) + 0j

    x = np.fft.irfft(Xf, n=T)

    f_rfft = np.fft.rfftfreq(T, d=1.0 / fs)
    df = f_rfft[1] - f_rfft[0] if f_rfft.size > 1 else fs
    var_target = float(np.sum(np.maximum(psd_rfft, 0.0)) * df)
    var_current = float(np.mean(x ** 2))

    if var_current > 0 and var_target > 0:
        x *= np.sqrt(var_target / var_current)

    return x.astype(np.float32)


def make_freq_grid_from_welch(cfg: Config) -> np.ndarray:
    dummy = np.zeros(cfg.T, dtype=np.float32)
    f, _ = signal.csd(
        dummy,
        dummy,
        fs=cfg.fs,
        window=cfg.window,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        detrend=cfg.detrend,
        scaling=cfg.scaling,
    )
    return f.astype(np.float32)


def compute_psd_welch(x: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    f, Pxx = signal.csd(
        x,
        x,
        fs=cfg.fs,
        window=cfg.window,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        detrend=cfg.detrend,
        scaling=cfg.scaling,
    )
    Pxx = np.real(Pxx)
    Pxx = np.maximum(Pxx, cfg.min_noise_floor_v2hz)
    return f.astype(np.float32), Pxx.astype(np.float32)


def interp_psd_to_rfft_grid(
    f_src: np.ndarray,
    psd_src: np.ndarray,
    f_dst: np.ndarray,
    floor: float,
) -> np.ndarray:
    y = np.interp(f_dst, f_src, psd_src, left=psd_src[0], right=psd_src[-1])
    return np.maximum(y, floor).astype(np.float32)


def make_amplitude_modulated_series(
    rng: np.random.Generator,
    x: np.ndarray,
    frac: float,
) -> np.ndarray:
    if frac <= 0:
        return x.astype(np.float32)

    n = x.size
    ctrl_n = max(8, n // 1024)
    ctrl = 1.0 + frac * rng.standard_normal(ctrl_n)
    ctrl = np.clip(ctrl, 0.6, 1.4)

    grid_ctrl = np.linspace(0, n - 1, ctrl_n)
    grid_full = np.arange(n)
    env = np.interp(grid_full, grid_ctrl, ctrl)
    return (x * env).astype(np.float32)


def make_noisy_sample_from_time_domain(
    rng: np.random.Generator,
    cfg: Config,
    f_welch: np.ndarray,
    f_rfft: np.ndarray,
    y_latent_welch: np.ndarray,
    x_latent: np.ndarray,
    peak_snr_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    x_obs
    y_obs
    y_target
    y_residual_total   where y_obs = y_target + residual approximately in PSD space
    y_obs_floor_model  smooth observation floor model used to synthesize time-domain noise
    """
    peak_value = float(np.max(y_latent_welch))
    nominal_floor = floor_level_from_peak_snr(peak_value, peak_snr_db, eps=cfg.eps)
    nominal_floor = max(nominal_floor, cfg.min_noise_floor_v2hz) * cfg.obs_floor_gain

    y_obs_floor_model = make_smooth_floor_profile(
        rng=rng,
        f=f_welch,
        nominal_floor=nominal_floor,
        min_noise_floor=cfg.min_noise_floor_v2hz,
        tilt_frac=cfg.floor_tilt_frac,
        wobble_frac=cfg.floor_wobble_frac,
        lf_excess_frac=cfg.lf_excess_frac,
        lf_excess_fc_hz=cfg.lf_excess_fc_hz,
        lf_excess_power=cfg.lf_excess_power,
        eps=cfg.eps,
    )
    y_target_floor_model = make_target_floor(rng, f_welch, y_obs_floor_model, cfg)

    y_obs_floor_rfft = interp_psd_to_rfft_grid(
        f_src=f_welch,
        psd_src=y_obs_floor_model,
        f_dst=f_rfft,
        floor=cfg.min_noise_floor_v2hz,
    )
    y_tgt_floor_rfft = interp_psd_to_rfft_grid(
        f_src=f_welch,
        psd_src=y_target_floor_model,
        f_dst=f_rfft,
        floor=cfg.min_noise_floor_v2hz,
    )

    obs_gain = rng.uniform(cfg.obs_noise_extra_gain_min, cfg.obs_noise_extra_gain_max)
    tgt_gain = rng.uniform(cfg.tgt_noise_extra_gain_min, cfg.tgt_noise_extra_gain_max)

    x_obs_floor = make_time_series_from_one_sided_psd(
        rng, cfg.fs, cfg.T, obs_gain * y_obs_floor_rfft, eps=cfg.eps
    )
    x_tgt_floor = make_time_series_from_one_sided_psd(
        rng, cfg.fs, cfg.T, tgt_gain * y_tgt_floor_rfft, eps=cfg.eps
    )

    x_latent_mod = make_amplitude_modulated_series(rng, x_latent, cfg.amplitude_jitter_frac)

    x_obs = x_latent_mod + x_obs_floor
    x_target_ts = x_latent_mod + x_tgt_floor

    _, y_obs = compute_psd_welch(x_obs, cfg)
    _, y_target = compute_psd_welch(x_target_ts, cfg)

    y_target = np.minimum(y_target, y_obs)
    y_target = np.maximum(y_target, cfg.min_noise_floor_v2hz)

    y_residual_total = np.maximum(y_obs - y_target, 0.0).astype(np.float32)
    return (
        x_obs.astype(np.float32),
        y_obs.astype(np.float32),
        y_target.astype(np.float32),
        y_residual_total,
        y_obs_floor_model.astype(np.float32),
    )


def plot_examples(
    out_dir: Path,
    tag: str,
    fs: float,
    X: np.ndarray,
    f: np.ndarray,
    Y_signal: np.ndarray,
    Y_target: np.ndarray,
    Y_obs: np.ndarray,
    Y_floor: np.ndarray,
    params: np.ndarray,
    max_time_plot_samples: int,
    n_examples: int,
    rng: np.random.Generator,
    plot_max_freq_hz: Optional[float] = None,
) -> None:
    ensure_dir(out_dir)
    N = X.shape[0]
    idx = rng.choice(N, size=min(n_examples, N), replace=False)

    plt.figure(figsize=(12, 7))
    for k, i in enumerate(idx, start=1):
        t = np.arange(min(max_time_plot_samples, X.shape[1])) / fs
        snr_val = params[i, 3]
        snr_str = "clean" if int(params[i, 4]) == 0 else f"{snr_val:.1f} dB"
        plt.plot(t, X[i, :t.size], label=f"ex {i} (noisy={int(params[i,4])}, peak_snr={snr_str})")
        if k >= 6:
            break
    plt.xlabel("t (s)")
    plt.ylabel("Amplitude (arb.)")
    plt.title(f"{tag}: example synthetic time series")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_time_examples.png", dpi=200)
    plt.close()

    for i in idx:
        y_sig = np.asarray(Y_signal[i], dtype=np.float64)
        y_obs = np.asarray(Y_obs[i], dtype=np.float64)
        y_tgt = np.asarray(Y_target[i], dtype=np.float64)
        y_floor = np.asarray(Y_floor[i], dtype=np.float64)

        f0_i = float(params[i, 0])
        df_i = float(params[i, 1])
        amp_i = float(params[i, 2])
        snr_val = params[i, 3]
        is_noisy = int(params[i, 4])
        snr_str = "clean" if is_noisy == 0 else f"{snr_val:.1f} dB"

        plt.figure(figsize=(11, 6))
        plt.plot(f, y_obs, label="Y_obs", linewidth=1.6)
        plt.plot(f, y_tgt, "--", label="Y_target (train target)", linewidth=1.8)
        plt.plot(f, y_sig, "-.", label="Y_signal only", linewidth=1.4, alpha=0.9)
        plt.plot(f, y_floor, ":", label="residual to remove", linewidth=1.8)

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
            plt.plot(f[mask], y_obs[mask], label="Y_obs", linewidth=1.8)
            plt.plot(f[mask], y_tgt[mask], "--", label="Y_target (train target)", linewidth=2.0)
            plt.plot(f[mask], y_sig[mask], "-.", label="Y_signal only", linewidth=1.5, alpha=0.9)
            plt.plot(f[mask], y_floor[mask], ":", label="residual to remove", linewidth=1.8)
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
    Y_signal: np.ndarray,
    Y_target: np.ndarray,
    Y_obs: np.ndarray,
    Y_floor: np.ndarray,
    Y_obs_floor: np.ndarray,
    params: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        f=f.astype(np.float32),
        Y_signal=Y_signal.astype(np.float32),
        Y_target=Y_target.astype(np.float32),
        Y_tilde=Y_target.astype(np.float32),   # compatibility alias
        Y_obs=Y_obs.astype(np.float32),
        Y_floor=Y_floor.astype(np.float32),
        Y_obs_floor=Y_obs_floor.astype(np.float32),
        params=params.astype(np.float32),
        meta_json=json.dumps(meta),
    )


def build_dataset(
    cfg: Config,
    rng: np.random.Generator,
    N: int,
    noisy: bool,
    snr_db_choices: Optional[Tuple[float, ...]],
    split_name: str,
):
    f = make_freq_grid_from_welch(cfg)
    F = f.size
    f_rfft = np.fft.rfftfreq(cfg.T, d=1.0 / cfg.fs).astype(np.float32)

    f0 = rng.uniform(cfg.f0_min, cfg.f0_max, size=N).astype(np.float32)
    deltaf = rng.uniform(cfg.deltaf_min, cfg.deltaf_max, size=N).astype(np.float32)
    A = sample_log_uniform(rng, cfg.A_min, cfg.A_max, size=N).astype(np.float32)

    meta_extra: Dict[str, Any] = {
        "generation_mode": "time_domain_then_welch",
        "welch": {
            "fs": cfg.fs,
            "nperseg": cfg.nperseg,
            "noverlap": cfg.noverlap,
            "window": cfg.window,
            "detrend": cfg.detrend,
            "scaling": cfg.scaling,
        },
    }

    if noisy:
        if not snr_db_choices:
            raise ValueError("Noisy dataset requested but snr_db_choices is empty.")
        peak_snr_db = rng.choice(np.array(snr_db_choices, dtype=np.float32), size=N, replace=True)
        is_noisy = np.ones(N, dtype=np.float32)
        meta_extra.update(
            {
                "noise_model": "time_domain_obs_and_target_with_welch",
                "snr_db_choices": [float(v) for v in snr_db_choices],
            }
        )
    else:
        peak_snr_db = np.full(N, np.inf, dtype=np.float32)
        is_noisy = np.zeros(N, dtype=np.float32)
        meta_extra.update({"noise_model": "clean_signal_through_welch"})

    X = np.zeros((N, cfg.T), dtype=np.float32)
    Y_signal = np.zeros((N, F), dtype=np.float32)
    Y_target = np.zeros((N, F), dtype=np.float32)
    Y_obs = np.zeros((N, F), dtype=np.float32)
    Y_floor = np.zeros((N, F), dtype=np.float32)
    Y_obs_floor = np.zeros((N, F), dtype=np.float32)

    for i in range(N):
        y_main_welch = gaussian_psd(f, float(f0[i]), float(deltaf[i]), float(A[i])).astype(np.float32)
        y_shoulder_welch = maybe_make_shoulder(rng, f, float(f0[i]), float(deltaf[i]), float(A[i]), cfg)
        y_latent_welch = np.maximum(y_main_welch + y_shoulder_welch, cfg.min_noise_floor_v2hz)

        y_latent_rfft = interp_psd_to_rfft_grid(
            f_src=f,
            psd_src=y_latent_welch,
            f_dst=f_rfft,
            floor=cfg.min_noise_floor_v2hz,
        )
        x_latent = make_time_series_from_one_sided_psd(rng, cfg.fs, cfg.T, y_latent_rfft, eps=cfg.eps)

        # Store signal-only PSD after Welch, not idealized analytic Gaussian only.
        _, y_signal_i = compute_psd_welch(x_latent, cfg)

        if noisy:
            x_obs_i, y_obs_i, y_target_i, y_resid_i, y_obs_floor_i = make_noisy_sample_from_time_domain(
                rng=rng,
                cfg=cfg,
                f_welch=f,
                f_rfft=f_rfft,
                y_latent_welch=y_signal_i,
                x_latent=x_latent,
                peak_snr_db=float(peak_snr_db[i]),
            )
        else:
            x_obs_i = x_latent.copy()
            y_obs_i = y_signal_i.copy()
            y_target_i = y_signal_i.copy()
            y_resid_i = np.zeros_like(y_signal_i, dtype=np.float32)
            y_obs_floor_i = np.full_like(y_signal_i, cfg.min_noise_floor_v2hz, dtype=np.float32)

        X[i] = x_obs_i
        Y_signal[i] = y_signal_i
        Y_target[i] = y_target_i
        Y_obs[i] = y_obs_i
        Y_floor[i] = y_resid_i
        Y_obs_floor[i] = y_obs_floor_i

        if (i + 1) % 500 == 0:
            print(f"[{split_name}] generated {i+1}/{N}")

    params = np.column_stack([f0, deltaf, A, peak_snr_db.astype(np.float32), is_noisy]).astype(np.float32)
    return X, f, Y_signal, Y_target, Y_obs, Y_floor, Y_obs_floor, params, meta_extra


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    run_id = cfg.run_id.strip() if cfg.run_id.strip() else make_run_id(cfg)
    run_dir = Path(cfg.runs_root) / run_id
    datasets_dir = run_dir / "datasets"
    plots_dir = run_dir / "plots"
    ensure_dir(datasets_dir)
    ensure_dir(plots_dir)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    written: Dict[str, Path] = {}

    print("\n=== Building train_gen1 (clean-only) ===")
    X1, f, Ys1, Yt1, Yo1, Yf1, Yof1, p1, meta1 = build_dataset(
        cfg=cfg, rng=rng, N=cfg.N_gen1, noisy=False, snr_db_choices=None, split_name="train_gen1"
    )
    p_gen1 = datasets_dir / "train_gen1_clean.npz"
    save_dataset_npz(p_gen1, X1, f, Ys1, Yt1, Yo1, Yf1, Yof1, p1, {"split": "train_gen1", "noisy": False, **meta1})
    plot_examples(
        plots_dir, "train_gen1_clean", cfg.fs, X1, f, Ys1, Yt1, Yo1, Yf1, p1,
        cfg.max_time_plot_samples, cfg.n_plot_examples, rng, cfg.plot_max_freq_hz
    )
    written["train_gen1_clean"] = p_gen1

    for snr_db in cfg.GEN2_SNR_DB_LIST:
        print(f"\n=== Building train_gen2 (noisy) @ peak_snr_db={snr_db:.1f} dB ===")
        X2, f2, Ys2, Yt2, Yo2, Yf2, Yof2, p2, meta2 = build_dataset(
            cfg=cfg, rng=rng, N=cfg.N_gen2_per_snr, noisy=True,
            snr_db_choices=(float(snr_db),), split_name=f"train_gen2_snr{snr_db:.1f}dB"
        )
        assert np.allclose(f2, f)
        fname = f"train_gen2_snr{_snr_piece(float(snr_db))}dB.npz"
        p_gen2 = datasets_dir / fname
        save_dataset_npz(
            p_gen2, X2, f, Ys2, Yt2, Yo2, Yf2, Yof2, p2,
            {"split": "train_gen2", "noisy": True, "peak_snr_db_bucket": float(snr_db), **meta2}
        )
        plot_examples(
            plots_dir, f"train_gen2_snr{_snr_piece(float(snr_db))}dB",
            cfg.fs, X2, f, Ys2, Yt2, Yo2, Yf2, p2,
            cfg.max_time_plot_samples, cfg.n_plot_examples, rng, cfg.plot_max_freq_hz
        )
        written[f"train_gen2_snr{snr_db:.1f}dB"] = p_gen2

    print("\n=== Building train_gen3_mix (50% clean / 50% noisy) ===")
    N3_clean = cfg.N_gen3 // 2
    N3_noisy = cfg.N_gen3 - N3_clean
    X3c, f3, Ys3c, Yt3c, Yo3c, Yf3c, Yof3c, p3c, _ = build_dataset(
        cfg=cfg, rng=rng, N=N3_clean, noisy=False, snr_db_choices=None, split_name="train_gen3_clean_half"
    )
    X3n, f3b, Ys3n, Yt3n, Yo3n, Yf3n, Yof3n, p3n, meta3n = build_dataset(
        cfg=cfg, rng=rng, N=N3_noisy, noisy=True, snr_db_choices=cfg.GEN3_SNR_DB_LIST,
        split_name="train_gen3_noisy_half"
    )
    assert np.allclose(f3, f3b)

    X3 = np.concatenate([X3c, X3n], axis=0)
    Ys3 = np.concatenate([Ys3c, Ys3n], axis=0)
    Yt3 = np.concatenate([Yt3c, Yt3n], axis=0)
    Yo3 = np.concatenate([Yo3c, Yo3n], axis=0)
    Yf3 = np.concatenate([Yf3c, Yf3n], axis=0)
    Yof3 = np.concatenate([Yof3c, Yof3n], axis=0)
    p3 = np.concatenate([p3c, p3n], axis=0)

    perm = rng.permutation(cfg.N_gen3)
    X3, Ys3, Yt3, Yo3, Yf3, Yof3, p3 = X3[perm], Ys3[perm], Yt3[perm], Yo3[perm], Yf3[perm], Yof3[perm], p3[perm]

    p_gen3 = datasets_dir / "train_gen3_mix_50_50.npz"
    save_dataset_npz(
        p_gen3, X3, f3, Ys3, Yt3, Yo3, Yf3, Yof3, p3,
        {"split": "train_gen3", "noisy": "mixed_50_50", "gen3_snr_db_choices": list(cfg.GEN3_SNR_DB_LIST), **meta3n}
    )
    plot_examples(
        plots_dir, "train_gen3_mix_50_50", cfg.fs, X3, f3, Ys3, Yt3, Yo3, Yf3, p3,
        cfg.max_time_plot_samples, cfg.n_plot_examples, rng, cfg.plot_max_freq_hz
    )
    written["train_gen3_mix_50_50"] = p_gen3

    print("\n=== Building eval (held-out) ===")
    Xe, fe, Yse, Yte, Yoe, Yfe, Yofe, pe, metae = build_dataset(
        cfg=cfg, rng=rng, N=cfg.N_eval, noisy=True, snr_db_choices=cfg.EVAL_SNR_DB_LIST, split_name="eval"
    )
    p_eval = datasets_dir / "eval_mixed_snr.npz"
    save_dataset_npz(
        p_eval, Xe, fe, Yse, Yte, Yoe, Yfe, Yofe, pe,
        {"split": "eval", "snr_db_choices": list(cfg.EVAL_SNR_DB_LIST), **metae}
    )
    plot_examples(
        plots_dir, "eval_mixed_snr", cfg.fs, Xe, fe, Yse, Yte, Yoe, Yfe, pe,
        cfg.max_time_plot_samples, cfg.n_plot_examples, rng, cfg.plot_max_freq_hz
    )
    written["eval_mixed_snr"] = p_eval

    write_manifest(run_dir, cfg, written)

    print("\nDone.")
    print(f"RUN_ID:          {run_id}")
    print(f"Run folder:      {run_dir.resolve()}")
    print(f"Datasets folder: {datasets_dir.resolve()}")
    print(f"Plots folder:    {plots_dir.resolve()}")
    print(f"Manifest:        {(run_dir / 'manifest.json').resolve()}")

    err = np.max(np.abs(Yoe[0] - (Yte[0] + Yfe[0])))
    print(f"Sanity check max|Y_obs - (Y_target + Y_floor)| on eval[0]: {err:.3e}")
    print("Note: this identity is no longer exact because both PSDs come from separate time-domain realizations + Welch.")


if __name__ == "__main__":
    main()