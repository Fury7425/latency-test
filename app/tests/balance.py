# tests/balance.py
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

def run(core: "AudioCore", log, freq=1000.0, tone_dur=1.0, settle=0.2):
    log("[BALANCE] Place mic on LEFT earcup, then click OK")
    messagebox.showinfo("Balance", "Place mic on LEFT earcup, then click OK")
    sig = core.generate_sine(freq=freq, duration=tone_dur)
    recL = _play_and_record(core, sig, both=True, settle=settle)

    log("[BALANCE] Move mic to RIGHT earcup, then click OK")
    messagebox.showinfo("Balance", "Move mic to RIGHT earcup, then click OK")
    core.generate_sine(freq=freq, duration=tone_dur)
    recR = _play_and_record(core, core.test_signal, both=True, settle=settle)

    levelL = dbfs(recL); levelR = dbfs(recR)
    diff = levelL - levelR

    res = TestResult(
        "balance",
        params={"freq": freq, "duration": tone_dur},
        metrics={"left_dBFS": levelL, "right_dBFS": levelR, "L_minus_R_dB": diff},
    )
    log(f"[BALANCE] L {levelL:.2f} dBFS, R {levelR:.2f} dBFS, Δ {diff:.2f} dB")
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
