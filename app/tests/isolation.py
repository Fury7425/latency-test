# tests/isolation.py
# robust relative/absolute imports
try:
    from ..core.audio import AudioCore
    from ..core.utils import dbfs
    from .base import TestResult
except ImportError:
    from core.audio import AudioCore
    from core.utils import dbfs
    from tests.base import TestResult

from tkinter import messagebox

def run(core: "AudioCore", log, noise_dur=2.0, amp=0.4):
    log("[ISOLATION] Inside: mic near earcup seal. OK to measure.")
    messagebox.showinfo("Isolation", "Place mic near earcup seal (inside). Click OK.")
    sig = core.generate_pink_noise(duration=noise_dur, amp=amp)
    rec_in = _play_and_record(core, sig, both=True, settle=0.05)

    log("[ISOLATION] Outside: mic ~5–10 cm outside the cup. OK to measure.")
    messagebox.showinfo("Isolation", "Move mic 5–10 cm outside the cup. Click OK.")
    core.generate_pink_noise(duration=noise_dur, amp=amp)
    rec_out = _play_and_record(core, core.test_signal, both=True, settle=0.05)

    in_db = dbfs(rec_in)
    out_db = dbfs(rec_out)
    delta = in_db - out_db

    res = TestResult(
        "isolation_inside_out",
        params={"noise_dur": noise_dur, "amp": amp},
        metrics={"inside_dBFS": in_db, "outside_dBFS": out_db, "delta_dB": delta},
    )
    log(f"[ISOLATION] Inside {in_db:.2f} dBFS, Outside {out_db:.2f} dBFS, Δ {delta:.2f} dB")
    return res

def _play_and_record(core, mono_signal, both=False, settle=0.05, rec_dur=None):
    import threading, time
    if rec_dur is None:
        rec_dur = float(core.duration) + 0.3
    rec = {"audio": None}

    def worker():
        rec["audio"] = core.record_audio(rec_dur)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.05 + settle)

    if both:
        core.play_stereo(mono_signal, mono_signal)
    else:
        core.play_mono(mono_signal)

    t.join()
    return rec["audio"]
