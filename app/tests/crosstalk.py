# tests/crosstalk.py
# robust relative/absolute imports
try:
    from ..core.audio import AudioCore
    from ..core.utils import rms
    from .base import TestResult
except ImportError:
    from core.audio import AudioCore
    from core.utils import rms
    from tests.base import TestResult

from tkinter import messagebox
import numpy as np

def run(core: "AudioCore", log, freq=1000.0, tone_dur=1.0, settle=0.2, direction="LtoR"):
    if direction == "LtoR":
        log("[CROSSTALK] Mic on LEFT (PRIMARY), OK")
        messagebox.showinfo("Crosstalk", "Place mic on LEFT earcup (PRIMARY), then click OK")
        sig = core.generate_sine(freq=freq, duration=tone_dur)
        rec_primary = _play_and_record(core, sig, left_only=True, settle=settle)

        log("[CROSSTALK] Mic on RIGHT (LEAK), OK")
        messagebox.showinfo("Crosstalk", "Move mic to RIGHT earcup (LEAK), then click OK")
        core.generate_sine(freq=freq, duration=tone_dur)
        rec_leak = _play_and_record(core, core.test_signal, left_only=True, settle=settle)
    else:
        log("[CROSSTALK] Mic on RIGHT (PRIMARY), OK")
        messagebox.showinfo("Crosstalk", "Place mic on RIGHT earcup (PRIMARY), then click OK")
        sig = core.generate_sine(freq=freq, duration=tone_dur)
        rec_primary = _play_and_record(core, sig, right_only=True, settle=settle)

        log("[CROSSTALK] Mic on LEFT (LEAK), OK")
        messagebox.showinfo("Crosstalk", "Move mic to LEFT earcup (LEAK), then click OK")
        core.generate_sine(freq=freq, duration=tone_dur)
        rec_leak = _play_and_record(core, core.test_signal, right_only=True, settle=settle)

    p = rms(rec_primary); l = rms(rec_leak)
    crosstalk_db = 20 * np.log10(max(l, 1e-12) / max(p, 1e-12))

    res = TestResult(
        "crosstalk",
        params={"freq": freq, "duration": tone_dur, "direction": direction},
        metrics={"primary_rms": p, "leak_rms": l, "crosstalk_dB": crosstalk_db},
    )
    log(f"[CROSSTALK {direction}] {crosstalk_db:.2f} dB")
    return res

def _play_and_record(core, mono_signal, left_only=False, right_only=False, both=False, settle=0.05, rec_dur=None):
    import threading, time
    if rec_dur is None:
        rec_dur = float(core.duration) + 0.3
    rec = {"audio": None}

    def worker():
        rec["audio"] = core.record_audio(rec_dur)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.05 + settle)

    if left_only:
        core.play_stereo(mono_signal, np.zeros_like(mono_signal))
    elif right_only:
        core.play_stereo(np.zeros_like(mono_signal), mono_signal)
    else:
        core.play_mono(mono_signal)

    t.join()
    return rec["audio"]
