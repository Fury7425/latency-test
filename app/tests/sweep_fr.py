# app/tests/sweep_fr.py
import os, datetime, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from ..core.audio import AudioCore
from ..core.utils import ensure_dir
from .base import TestResult

def _single_sweep(core, f0=20, f1=20000, duration=6.0, amp=0.5):
    """Run ONE sweep, return (grid_Hz, mag_db, delay_ms)."""
    sig, t = core.generate_log_chirp(f0=f0, f1=f1, duration=duration, amp=amp)
    # record slightly longer than sweep to catch tail
    rec = _play_and_record(core, sig, both=True, settle=0.05, rec_dur=duration+0.6)

    # crude alignment using cross-correlation
    delay_ms, _ = AudioCore.find_delay_ms(rec, sig, core.sample_rate)
    delay_s = 0.0 if delay_ms is None else delay_ms/1000.0
    shift = int(round(delay_s * core.sample_rate))
    rec_aligned = rec[shift: shift + len(sig)] if shift >= 0 else rec[:len(sig)]
    if len(rec_aligned) < len(sig):
        rec_aligned = np.pad(rec_aligned, (0, len(sig)-len(rec_aligned)))

    # envelope of recorded sweep
    env = np.abs(signal.hilbert(rec_aligned))
    env = env / (np.max(env)+1e-12)

    # map time→frequency for a log sweep
    T = duration
    tvec = np.linspace(0, duration, len(sig), endpoint=False)
    freqs_inst = f0 * (f1/f0) ** (tvec / T)

    # sample envelope on a fixed log grid
    grid = np.logspace(np.log10(max(20,f0)), np.log10(min(f1,20000)), 200)
    mag = np.zeros_like(grid)
    for i, f in enumerate(grid):
        idx = np.argmin(np.abs(freqs_inst - f))
        mag[i] = env[idx]

    mag_db = 20*np.log10(np.maximum(mag, 1e-6))
    return grid, mag_db, delay_ms

def run(core, log, f0=20, f1=20000, duration=6.0, repeats=3, save_plot_dir=None, show_std=True):
    """
    Run the sweep multiple times and save 2 plots:
      - overlay of all runs
      - average (and ±1σ if show_std=True)
    Returns a TestResult with all curves + average.
    """
    log(f"[SWEEP FR] {repeats} runs, {f0}–{f1} Hz, {duration:.1f}s each")
    curves = []
    delays = []

    for i in range(repeats):
        grid, mag_db, dms = _single_sweep(core, f0=f0, f1=f1, duration=duration)
        curves.append(mag_db); delays.append(dms)
        log(f"  ✓ run {i+1}/{repeats}: delay {dms:.1f} ms")

    curves = np.stack(curves, axis=0)              # [repeats, 200]
    mean_db = np.mean(curves, axis=0)
    std_db  = np.std(curves, axis=0)

    files = {}
    if save_plot_dir:
        ensure_dir(save_plot_dir)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1) overlay
        overlay_path = os.path.join(save_plot_dir, f"sweep_fr_overlay_{ts}.png")
        plt.figure(figsize=(9,5))
        for r in range(curves.shape[0]):
            plt.semilogx(grid, curves[r], alpha=0.65)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("Relative Level (dB)")
        plt.title("Relative Frequency Response — All Runs")
        plt.grid(True, which='both', ls=':', alpha=0.6)
        plt.tight_layout(); plt.savefig(overlay_path); plt.close()
        files["overlay_plot"] = overlay_path
        log(f"  saved -> {overlay_path}")

        # 2) average (+/- 1σ)
        avg_path = os.path.join(save_plot_dir, f"sweep_fr_average_{ts}.png")
        plt.figure(figsize=(9,5))
        plt.semilogx(grid, mean_db, label="Average", linewidth=2)
        if show_std and repeats > 1:
            plt.fill_between(grid, mean_db-std_db, mean_db+std_db, alpha=0.2, label="±1σ")
        plt.xlabel("Frequency (Hz)"); plt.ylabel("Relative Level (dB)")
        plt.title("Relative Frequency Response — Average")
        plt.grid(True, which='both', ls=':', alpha=0.6)
        plt.legend()
        plt.tight_layout(); plt.savefig(avg_path); plt.close()
        files["average_plot"] = avg_path
        log(f"  saved -> {avg_path}")

    res = TestResult(
        "sweep_fr_multi",
        params={"f0": f0, "f1": f1, "duration": duration, "repeats": repeats},
        metrics={
            "delays_ms": delays,
            "delay_avg_ms": float(np.mean(delays)),
            "delay_std_ms": float(np.std(delays)) if len(delays) > 1 else 0.0
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

def _play_and_record(core, mono_signal, left_only=False, right_only=False, both=False, settle=0.05, rec_dur=None):
    import threading, time, numpy as np
    if rec_dur is None: rec_dur = float(core.duration) + 0.3
    rec = {"audio": None}
    def worker(): rec["audio"] = core.record_audio(rec_dur)
    t = threading.Thread(target=worker, daemon=True); t.start()
    time.sleep(0.05 + settle)
    if both:
        core.play_stereo(mono_signal, mono_signal)
    else:
        core.play_mono(mono_signal)
    t.join(); return rec["audio"]
