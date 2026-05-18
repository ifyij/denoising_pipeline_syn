# 01_synth_make_datasets.py
"""
Synthetic dataset generator for CNN mapping: X(t) [V] -> Y_tilde(f) [V^2/Hz]

Outputs (per dataset):
- X: time series in volts, shape (N, T)
- f: frequency axis for PSD targets (Welch freqs), shape (F,)
- Y_tilde: clean Gaussian PSD targets, shape (N, F)
- Y_obs: observed PSD from Welch on X (noisy or clean), shape (N, F)
- params: (f0, deltaf, A, snr_db, is_noisy) per sample

Generates:
- train_gen1 (clean-only)
- train_gen2_snrXX (one per SNR in GEN2_SNR_DB_LIST)
- train_gen3_mix (50% clean, 50% noisy)
- eval (held-out; can be mixed or fixed SNR list)

Creates folders automatically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


# -----------------------------
# Configuration (edit here)
# -----------------------------

@dataclass
class Config:
    # Repro
    seed: int = 123

    # Output root
    out_root: str = "synthetic_datasets"

    # Time series
    fs: float = 1e6          # Hz (edit to match CECE sampling rate)
    T: int = 65536           # samples per time series (power-of-two is convenient)

    # Welch PSD settings (used for targets + observed PSD)
    nperseg: int = 4096
    noverlap: int = 2048
    window: str = "hann"
    detrend: str = "constant"
    scaling: str = "density"  # ensures V^2/Hz units

    # Parameter ranges for Gaussian PSD targets
    # NOTE: ensure f0 range is within [0, fs/2]
    f0_min: float = 5e3
    f0_max: float = 120e3
    deltaf_min: float = 1e3
    deltaf_max: float = 30e3
    #decrease this range
    A_min: float = 1e-10      # V^2/Hz (edit to plausible magnitude for your data)
    A_max: float = 1e-7

    #set order of magnitude of noise power 

    # Dataset sizes
    N_gen1: int = 8000
    N_gen2_per_snr: int = 4000
    N_gen3: int = 8000
    N_eval: int = 2000

    # Gen2 SNRs (dB): "a couple of noisy datasets"
    GEN2_SNR_DB_LIST: Tuple[float, ...] = (20.0, 10.0)

    # Gen3 mix: noisy half uses this SNR list (sampled uniformly)
    GEN3_NOISY_SNR_DB_LIST: Tuple[float, ...] = (25.0, 15.0, 10.0, 7.0)

    # Eval: you can choose to mix SNRs or set fixed list
    EVAL_SNR_DB_LIST: Tuple[float, ...] = (30.0, 20.0, 15.0, 10.0, 7.0, 5.0)

    # Sanity plots
    n_plot_examples: int = 3        # number of random examples to plot per dataset
    max_time_plot_samples: int = 4000  # plot first N samples in time-domain figures

    # Small epsilon to avoid sqrt(0)
    eps: float = 1e-30


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sample_log_uniform(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    """Sample log-uniform between lo and hi (positive)."""
    if lo <= 0 or hi <= 0:
        raise ValueError("Log-uniform requires positive bounds.")
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def gaussian_psd(f: np.ndarray, f0: float, deltaf: float, A: float) -> np.ndarray:
    """One-sided Gaussian PSD (V^2/Hz) on frequency grid f."""
    return A * np.exp(-0.5 * ((f - f0) / deltaf) ** 2)


def make_time_series_from_psd(
    rng: np.random.Generator,
    fs: float,
    T: int,
    target_psd_rfft: np.ndarray,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Generate a real-valued time series x(t) whose spectrum is shaped by target_psd_rfft.
    This is an approximate construction: we create random complex rFFT bins, shape them
    by sqrt(PSD), irFFT back to time domain, then rescale to match target variance.

    target_psd_rfft is defined on rFFT freqs: rfftfreq(T, 1/fs) (length T//2+1).
    """
    # Random complex coefficients for rFFT bins
    z = rng.standard_normal(target_psd_rfft.size) + 1j * rng.standard_normal(target_psd_rfft.size)

    # Shape by sqrt(PSD)
    shape = np.sqrt(np.maximum(target_psd_rfft, 0.0) + eps)
    Xf = z * shape

    # Force DC and Nyquist to be real to keep time series real-valued
    Xf[0] = np.real(Xf[0]) + 0j
    if T % 2 == 0:
        Xf[-1] = np.real(Xf[-1]) + 0j

    # Back to time domain
    x = np.fft.irfft(Xf, n=T)

    # Rescale to match variance implied by one-sided PSD: var ≈ ∫ PSD df
    f_rfft = np.fft.rfftfreq(T, d=1.0 / fs)
    df = f_rfft[1] - f_rfft[0] if f_rfft.size > 1 else fs

    var_target = float(np.sum(target_psd_rfft) * df)  # one-sided approximation
    var_current = float(np.mean(x**2))

    if var_current > 0 and var_target > 0:
        x *= np.sqrt(var_target / var_current)

    return x.astype(np.float32)

#TODO CHANGE NOISE FORMULA 
def add_awgn_for_snr(
    rng: np.random.Generator,
    x: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Add white Gaussian noise to achieve SNR in time-domain power:
      SNR_dB = 10*log10(P_signal / P_noise), P = mean(x^2)
    """
    p_signal = float(np.mean(x**2))
    snr_lin = 10.0 ** (snr_db / 10.0)
    p_noise = p_signal / snr_lin if snr_lin > 0 else p_signal
    noise_std = np.sqrt(max(p_noise, 0.0))

    n = rng.standard_normal(x.shape).astype(np.float32) * noise_std
    return (x + n).astype(np.float32)


def compute_welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    window: str,
    detrend: str,
    scaling: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD for one time series."""
    f, Pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling=scaling,
        return_onesided=True,
    )
    return f.astype(np.float32), Pxx.astype(np.float32)


def plot_examples(
    out_dir: Path,
    tag: str,
    fs: float,
    X: np.ndarray,
    f: np.ndarray,
    Y_tilde: np.ndarray,
    Y_obs: np.ndarray,
    params: np.ndarray,
    max_time_plot_samples: int,
    n_examples: int,
    rng: np.random.Generator,
) -> None:
    """Save sanity-check plots for a dataset."""
    ensure_dir(out_dir)

    N = X.shape[0]
    idx = rng.choice(N, size=min(n_examples, N), replace=False)

    # Time series examples
    plt.figure(figsize=(12, 7))
    for k, i in enumerate(idx, start=1):
        t = np.arange(min(max_time_plot_samples, X.shape[1])) / fs
        plt.plot(t, X[i, :t.size], label=f"ex {i} (noisy={int(params[i,4])}, snr={params[i,3]:.1f} dB)")
        if k >= 6:
            break
    plt.xlabel("t (s)")
    plt.ylabel("Voltage (V)")
    plt.title(f"{tag}: example time series (first {min(max_time_plot_samples, X.shape[1])} samples)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_time_examples.png", dpi=200)
    plt.close()

    # PSD overlay examples
    plt.figure(figsize=(12, 7))
    for k, i in enumerate(idx, start=1):
        plt.plot(f, Y_obs[i], alpha=0.7, label=f"Y_obs ex {i}")
        plt.plot(f, Y_tilde[i], alpha=0.9, linestyle="--", label=f"Y_tilde ex {i}")
        if k >= 3:
            break
    plt.xlabel("f (Hz)")
    plt.ylabel(r"PSD (V$^2$/Hz)")
    plt.title(f"{tag}: PSD overlays (observed vs clean target)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_psd_overlays.png", dpi=200)
    plt.close()


def save_dataset_npz(
    out_path: Path,
    X: np.ndarray,
    f: np.ndarray,
    Y_tilde: np.ndarray,
    Y_obs: np.ndarray,
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
        params=params.astype(np.float32),
        meta_json=json.dumps(meta),
    )


def build_dataset(
    cfg: Config,
    rng: np.random.Generator,
    N: int,
    noisy: bool,
    snr_db_choices: Tuple[float, ...] | None,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create dataset arrays:
      X: (N, T)
      f_w: (F,)
      Y_tilde: (N, F)
      Y_obs: (N, F)
      params: (N, 5) = [f0, deltaf, A, snr_db, is_noisy]
    """
    fs, T = cfg.fs, cfg.T

    # Welch freq grid determined by settings (compute once using dummy)
    f_w, _ = compute_welch_psd(
        x=np.zeros(T, dtype=np.float32),
        fs=fs,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        window=cfg.window,
        detrend=cfg.detrend,
        scaling=cfg.scaling,
    )
    F = f_w.size

    # rFFT grid for constructing time series
    f_r = np.fft.rfftfreq(T, d=1.0 / fs).astype(np.float32)

    # Sample Gaussian parameters
    f0 = rng.uniform(cfg.f0_min, cfg.f0_max, size=N).astype(np.float32)
    deltaf = rng.uniform(cfg.deltaf_min, cfg.deltaf_max, size=N).astype(np.float32)
    A = sample_log_uniform(rng, cfg.A_min, cfg.A_max, size=N).astype(np.float32)

    # Choose SNR per sample (if noisy)
    if noisy:
        if not snr_db_choices or len(snr_db_choices) == 0:
            raise ValueError("Noisy dataset requested but snr_db_choices is empty.")
        snr_db = rng.choice(np.array(snr_db_choices, dtype=np.float32), size=N, replace=True)
        is_noisy = np.ones(N, dtype=np.float32)
    else:
        snr_db = np.full(N, np.inf, dtype=np.float32)  # sentinel for clean
        is_noisy = np.zeros(N, dtype=np.float32)

    X = np.zeros((N, T), dtype=np.float32)
    Y_tilde = np.zeros((N, F), dtype=np.float32)
    Y_obs = np.zeros((N, F), dtype=np.float32)

    # Generate samples
    for i in range(N):
        # Clean target PSD on both grids
        ytilde_r = gaussian_psd(f_r, float(f0[i]), float(deltaf[i]), float(A[i])).astype(np.float32)
        ytilde_w = gaussian_psd(f_w, float(f0[i]), float(deltaf[i]), float(A[i])).astype(np.float32)

        # Build clean time series consistent with target PSD
        x_clean = make_time_series_from_psd(rng, fs, T, ytilde_r, eps=cfg.eps)

        # Add noise if requested
        if noisy:
            x_use = add_awgn_for_snr(rng, x_clean, float(snr_db[i]))
        else:
            x_use = x_clean

        # Observed PSD from Welch (what CECE would give you)
        _, pxx = compute_welch_psd(
            x_use,
            fs=fs,
            nperseg=cfg.nperseg,
            noverlap=cfg.noverlap,
            window=cfg.window,
            detrend=cfg.detrend,
            scaling=cfg.scaling,
        )

        X[i] = x_use
        Y_tilde[i] = ytilde_w
        Y_obs[i] = pxx

        if (i + 1) % 500 == 0:
            print(f"[{split_name}] generated {i+1}/{N}")

    params = np.column_stack([f0, deltaf, A, snr_db.astype(np.float32), is_noisy]).astype(np.float32)
    return X, f_w, Y_tilde, Y_obs, params


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    out_root = Path(cfg.out_root)
    ensure_dir(out_root)

    # Save config
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    plots_dir = out_root / "plots"
    ensure_dir(plots_dir)

    datasets_dir = out_root / "datasets"
    ensure_dir(datasets_dir)

    # -------------------------
    # Gen 1: clean-only train
    # -------------------------
    print("\n=== Building train_gen1 (clean-only) ===")
    X1, f_w, Yt1, Yo1, p1 = build_dataset(
        cfg=cfg,
        rng=rng,
        N=cfg.N_gen1,
        noisy=False,
        snr_db_choices=None,
        split_name="train_gen1",
    )
    save_dataset_npz(
        datasets_dir / "train_gen1_clean.npz",
        X=X1, f=f_w, Y_tilde=Yt1, Y_obs=Yo1, params=p1,
        meta={"split": "train_gen1", "noisy": False},
    )
    plot_examples(
        out_dir=plots_dir,
        tag="train_gen1_clean",
        fs=cfg.fs, X=X1, f=f_w, Y_tilde=Yt1, Y_obs=Yo1, params=p1,
        max_time_plot_samples=cfg.max_time_plot_samples,
        n_examples=cfg.n_plot_examples,
        rng=rng,
    )

    # ---------------------------------------
    # Gen 2: a couple of noisy training sets
    # ---------------------------------------
    for snr in cfg.GEN2_SNR_DB_LIST:
        print(f"\n=== Building train_gen2 (noisy) @ SNR={snr} dB ===")
        X2, f_w2, Yt2, Yo2, p2 = build_dataset(
            cfg=cfg,
            rng=rng,
            N=cfg.N_gen2_per_snr,
            noisy=True,
            snr_db_choices=(float(snr),),
            split_name=f"train_gen2_snr{snr:g}dB",
        )
        assert np.allclose(f_w2, f_w)
        fname = f"train_gen2_snr{snr:g}dB".replace(".", "p")
        save_dataset_npz(
            datasets_dir / fname,
            X=X2, f=f_w, Y_tilde=Yt2, Y_obs=Yo2, params=p2,
            meta={"split": "train_gen2", "noisy": True, "snr_db": float(snr)},
        )
        plot_examples(
            out_dir=plots_dir,
            tag=f"train_gen2_snr{snr:g}dB".replace(".", "p"),
            fs=cfg.fs, X=X2, f=f_w, Y_tilde=Yt2, Y_obs=Yo2, params=p2,
            max_time_plot_samples=cfg.max_time_plot_samples,
            n_examples=cfg.n_plot_examples,
            rng=rng,
        )

    # ---------------------------------------
    # Gen 3: mixed dataset (50% clean / 50% noisy)
    # ---------------------------------------
    print("\n=== Building train_gen3_mix (50% clean / 50% noisy) ===")
    N3 = cfg.N_gen3
    N3_clean = N3 // 2
    N3_noisy = N3 - N3_clean

    X3c, f_w3, Yt3c, Yo3c, p3c = build_dataset(
        cfg=cfg,
        rng=rng,
        N=N3_clean,
        noisy=False,
        snr_db_choices=None,
        split_name="train_gen3_clean_half",
    )
    X3n, f_w3b, Yt3n, Yo3n, p3n = build_dataset(
        cfg=cfg,
        rng=rng,
        N=N3_noisy,
        noisy=True,
        snr_db_choices=cfg.GEN3_NOISY_SNR_DB_LIST,
        split_name="train_gen3_noisy_half",
    )
    assert np.allclose(f_w3, f_w3b)

    X3 = np.concatenate([X3c, X3n], axis=0)
    Yt3 = np.concatenate([Yt3c, Yt3n], axis=0)
    Yo3 = np.concatenate([Yo3c, Yo3n], axis=0)
    p3 = np.concatenate([p3c, p3n], axis=0)

    # Shuffle mix
    perm = rng.permutation(N3)
    X3, Yt3, Yo3, p3 = X3[perm], Yt3[perm], Yo3[perm], p3[perm]

    save_dataset_npz(
        datasets_dir / "train_gen3_mix_50_50.npz",
        X=X3, f=f_w3, Y_tilde=Yt3, Y_obs=Yo3, params=p3,
        meta={"split": "train_gen3", "noisy": "mixed_50_50", "gen3_noisy_snr_choices": list(cfg.GEN3_NOISY_SNR_DB_LIST)},
    )
    plot_examples(
        out_dir=plots_dir,
        tag="train_gen3_mix_50_50",
        fs=cfg.fs, X=X3, f=f_w3, Y_tilde=Yt3, Y_obs=Yo3, params=p3,
        max_time_plot_samples=cfg.max_time_plot_samples,
        n_examples=cfg.n_plot_examples,
        rng=rng,
    )

    # -------------------------
    # Eval: held-out
    # -------------------------
    print("\n=== Building eval (held-out; mixed SNR list) ===")
    Xe, f_we, Yte, Yoe, pe = build_dataset(
        cfg=cfg,
        rng=rng,
        N=cfg.N_eval,
        noisy=True,  # eval typically includes noise; you can flip this if desired
        snr_db_choices=cfg.EVAL_SNR_DB_LIST,
        split_name="eval",
    )
    save_dataset_npz(
        datasets_dir / "eval_mixed_snr.npz",
        X=Xe, f=f_we, Y_tilde=Yte, Y_obs=Yoe, params=pe,
        meta={"split": "eval", "snr_db_choices": list(cfg.EVAL_SNR_DB_LIST)},
    )
    plot_examples(
        out_dir=plots_dir,
        tag="eval_mixed_snr",
        fs=cfg.fs, X=Xe, f=f_we, Y_tilde=Yte, Y_obs=Yoe, params=pe,
        max_time_plot_samples=cfg.max_time_plot_samples,
        n_examples=cfg.n_plot_examples,
        rng=rng,
    )

    print("\nDone.")
    print(f"Datasets saved in: {datasets_dir.resolve()}")
    print(f"Plots saved in:    {plots_dir.resolve()}")
    print(f"Config saved in:   {(out_root / 'config.json').resolve()}")


if __name__ == "__main__":
    main()
