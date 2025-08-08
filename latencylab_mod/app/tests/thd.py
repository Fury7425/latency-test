
import numpy as np
from scipy import signal
from .base import TestResult

def run(core, log, tones=(100, 1000, 6000), tone_dur=1.0, amp=0.6):
    log("[THD] " + ", ".join(str(x) for x in tones))
    items = []
    for f in tones:
        sig = core.generate_sine(freq=f, duration=tone_dur, amp=amp)
        rec = _play_and_record(core, sig, both=True, settle=0.05)
        thd = _compute_thd(rec, f, sr=core.sample_rate)
        items.append({"freq": f, "thd_percent": thd})
        log(f"  {f} Hz -> {thd:.3f}%")
    res = TestResult("thd", params={"tones": list(tones), "tone_dur": tone_dur}, metrics={"items": items})
    return res

def _play_and_record(core, mono_signal, both=False, settle=0.05, rec_dur=None):
    import threading, time
    if rec_dur is None: rec_dur = float(core.duration) + 0.3
    rec = {"audio": None}
    def worker(): rec["audio"] = core.record_audio(rec_dur)
    t = threading.Thread(target=worker, daemon=True); t.start()
    time.sleep(0.05 + settle)
    if both: core.play_stereo(mono_signal, mono_signal)
    else: core.play_mono(mono_signal)
    t.join(); return rec["audio"]

def _compute_thd(x, f0, sr, n_harm=10):
    N = int(2**np.ceil(np.log2(len(x)))); win = signal.windows.hann(len(x), sym=False)
    X = np.fft.rfft(x*win, n=N); freqs = np.fft.rfftfreq(N, 1/sr)
    def bin_mag(f):
        k = np.argmin(np.abs(freqs - f)); return np.abs(X[k])
    fund = bin_mag(f0); harm_power = 0.0
    for k in range(2, n_harm+1):
        fk = f0*k
        if fk > sr/2: break
        harm_power += bin_mag(fk)**2
    thd = (np.sqrt(harm_power)/(np.abs(fund)+1e-12))*100.0
    return float(thd)
