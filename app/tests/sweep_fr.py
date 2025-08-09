# app/tests/sweep_fr.py
import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless/pyinstaller builds
import matplotlib.pyplot as plt
from scipy import signal

# --- Robust imports: work whether you're using package layout (app/...) or flat files ---
try:
    from ..core.audio import AudioCore
    from ..core.utils import ensure_dir
    from .base import TestResult
except ImportError:
    from core.audio import AudioCore
    from core.utils import ensure_dir
    from tests.base import TestResult


def _single_sweep(core: "AudioCore", f0=20, f1=20000, duration=6.0, amp=0.5):
    """
    Run ONE logarithmic sweep and return (grid_Hz, mag_db, delay_ms).
    Uses Hilbert envelope + time→freq mapping to produce a relative FR curve.
    """
    # Generate sweep
    sig, t = core.generate_log_chirp(f0=f0, f1=f1, duration=duration, amp=amp)

    # Record slightly longer than sweep to catch tail
    rec = _play_and_record(core, sig, both=True, settle=0.05, rec_dur=duration + 0.6)

    # Align recorded to reference via cross-correlation (approx)
    delay_ms, _ = AudioCore.find_delay_ms(rec, sig, core.sample_rate)
    delay_s = 0.0 if delay_ms is None else delay_ms / 1000.0
    shift = int(round(delay_s * core.sample_rate))

    if shift >= 0:
        rec_aligned = rec[shift: shift + len(sig)]
    else:
        # negative shift means recorded started before reference; just truncate
        rec_aligned = rec[: len(sig)]

    if len(rec_aligned) < len(sig):
        rec_aligned = np.pad(rec_aligned, (0, len(sig) - len(rec_aligned)))

    # Envelope of recorded sweep
    env = np.abs(signal.hilbert(rec_aligned))
    env = env / (np.max(env) + 1e-12)

    # Map time → instantaneous frequency for log sweep
    T = duration
    tvec = np.linspace(0, duration, len(sig), endpoint=False)
    freqs_inst = f0 * (f1 / f0) ** (tvec / T)

    # Sample envelope on a fixed log grid for comparability
    grid = np.logspace(np.log10(max(20, f0)), np.log10(min(f1, 20000)), 200)
    mag = np.zeros_like(grid)
    for i, f in enumerate(grid):
        idx = np.argmin(np.abs(freqs_inst - f))
        mag[i] = env[idx]

    mag_db = 20 * np.log10(np.maximum(mag, 1e-6))
    return grid, mag_db, (0.0 if delay_ms is None else float(delay_ms))


def run(core: "AudioCore",
        log,
        f0=20,
        f1=20000,
        duration=6.0,
        repeats=3,
        save_plot_dir=None,
        show_std=True,
        csv_path: str | None = None):
    """
    Run the sweep multiple times and save TWO plots:
      1) Overlay of all runs
      2) Average curve (with optional ±1σ band)

    Returns a TestResult with:
      - metrics: delays per run, delay mean/std
      - data: frequency grid, all curves (dB), mean/std arrays
      - files: paths to overlay/average plots (and CSV if requested)

    Args:
      repeats: number of runs (>=1)
      save_plot_dir: directory for .png plots (optional)
      show_std: include ±1σ band on the average plot
      csv_path: if provided, writes CSV with grid + each run + mean + std
    """
    repeats = max(1, int(repeats))
    log(f"[SWEEP FR] {repeats} runs, {f0}–{f1} Hz, {duration:.1f}s each")

    curves = []
    delays = []

    for i in range(repeats):
        grid, mag_db, dms = _single_sweep(core, f0=f0, f1=f1, duration=duration)
        curves.append(mag_db)
        delays.append(dms)
        log(f"  ✓ run {i+1}/{repeats}: delay {dms:.1f} ms")

    curves = np.stack(curves, axis=0)  # shape: [repeats, 200]
    mean_db = np.mean(curves, axis=0)
    std_db = np.std(curves, axis=0) if repeats > 1 else np.zeros_like(mean_db)

    files = {}
    if save_plot_dir:
        ensure_dir(save_plot_dir)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1) Overlay of all runs
        overlay_path = os.path.join(save_plot_dir, f"sweep_fr_overlay_{ts}.png")
        plt.figure(figsize=(9, 5))
        for r in range(curves.shape[0]):
            plt.semilogx(grid, curves[r], alpha=0.65)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Relative Level (dB)")
        plt.title("Relative Frequency Response — All Runs")
        plt.grid(True, which='both', ls=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(overlay_path)
        plt.close()
        files["overlay_plot"] = overlay_path
        log(f"  saved -> {overlay_path}")

        # 2) Average curve (+/- 1σ)
        avg_path = os.path.join(save_plot_dir, f"sweep_fr_average_{ts}.png")
        plt.figure(figsize=(9, 5))
        plt.semilogx(grid, mean_db, label="Average", linewidth=2)
        if show_std and repeats > 1:
            plt.fill_between(grid, mean_db - std_db, mean_db + std_db, alpha=0.2, label="±1σ")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Relative Level (dB)")
        plt.title("Relative Frequency Response — Average")
        plt.grid(True, which='both', ls=':', alpha=0.6)
        if show_std and repeats > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(avg_path)
        plt.close()
        files["average_plot"] = avg_path
        log(f"  saved -> {avg_path}")

    # Optional CSV export
    if csv_path:
        ensure_dir(os.path.dirname(csv_path) or ".")
        header = ["freq_Hz"] + [f"run_{i+1}_dB" for i in range(repeats)] + ["mean_dB", "std_dB"]
        data_mat = np.column_stack([grid, curves.T, mean_db, std_db])
        np.savetxt(csv_path, data_mat, delimiter=",", header=",".join(header), comments="", fmt="%.6f")
        files["csv"] = csv_path
        log(f"  saved -> {csv_path}")

    res = TestResult(
        "sweep_fr_multi",
        params={"f0": f0, "f1": f1, "duration": duration, "repeats": repeats},
        metrics={
            "delays_ms": [float(x) for x in delays],
            "delay_avg_ms": float(np.mean(delays)),
            "delay_std_ms": float(np.std(delays)) if repeats > 1 else 0.0
        },
        data={
            "freqs": grid.tolist(),
            "curves_db": curves.tolist(),
            "mean_db": mean_db.tolist(),
            "std_db": std_db.tolist()
        },
        files=files
    )
    log("[SWEEP FR] done.")
    return res


def _play_and_record(core: "AudioCore",
                     mono_signal,
                     left_only: bool = False,
                     right_only: bool = False,
                     both: bool = False,
                     settle: float = 0.05,
                     rec_dur: float | None = None):
    """
    Helper: record while playing the provided mono signal.
    If both=True, plays stereo (L=R=signal); else plays mono.
    """
    import threading
    import time

    if rec_dur is None:
        rec_dur = float(core.duration) + 0.3

    rec = {"audio": None}

    def worker():
        rec["audio"] = core.record_audio(rec_dur)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.05 + settle)

    if both:
        # Stereo play (L=R=signal)
        core.play_stereo(mono_signal, mono_signal)
    else:
        # Mono play
        core.play_mono(mono_signal)

    t.join()
    return rec["audio"]
