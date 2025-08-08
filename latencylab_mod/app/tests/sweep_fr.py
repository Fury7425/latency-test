
import os, datetime, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from ..core.audio import AudioCore
from ..core.utils import ensure_dir
from .base import TestResult

def run(core, log, f0=20, f1=20000, duration=6.0, save_plot_dir=None):
    log("[SWEEP FR] Keep mic steady near driver.")
    sig, t = core.generate_log_chirp(f0=f0, f1=f1, duration=duration, amp=0.5)
    rec = _play_and_record(core, sig, both=True, settle=0.05, rec_dur=duration+0.5)

    delay_ms, _ = AudioCore.find_delay_ms(rec, sig, core.sample_rate)
    delay_s = 0.0 if delay_ms is None else delay_ms/1000.0
    shift = int(round(delay_s * core.sample_rate))
    rec_aligned = rec[shift: shift + len(sig)] if shift >= 0 else rec[:len(sig)]
    if len(rec_aligned) < len(sig):
        rec_aligned = np.pad(rec_aligned, (0, len(sig)-len(rec_aligned)))

    env = np.abs(signal.hilbert(rec_aligned))
    env = env / (np.max(env)+1e-12)

    T = duration
    freqs_inst = f0 * (f1/f0) ** (t / T)

    grid = np.logspace(np.log10(max(20,f0)), np.log10(min(f1,20000)), 200)
    mag = np.zeros_like(grid)
    for i, f in enumerate(grid):
        idx = np.argmin(np.abs(freqs_inst - f)); mag[i] = env[idx]

    mag_db = 20*np.log10(np.maximum(mag, 1e-6))
    files = {}
    if save_plot_dir:
        ensure_dir(save_plot_dir)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(save_plot_dir, f"sweep_fr_{ts}.png")
        plt.figure(figsize=(9,5))
        plt.semilogx(grid, mag_db)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("Relative Level (dB)"); plt.title("Relative Frequency Response")
        plt.grid(True, which='both', ls=':', alpha=0.6)
        plt.tight_layout(); plt.savefig(out); plt.close()
        files["plot"] = out

    res = TestResult("sweep_fr",
        params={"f0": f0, "f1": f1, "duration": duration},
        metrics={"delay_ms": delay_ms},
        data={"freqs": grid.tolist(), "mag_db": mag_db.tolist()},
        files=files
    )
    log(f"[SWEEP FR] Done (delay {delay_ms:.1f} ms)")
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
