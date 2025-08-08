
import os, time, threading, datetime, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from ..core.utils import safe_mean, safe_std
from ..core.utils import ensure_dir
from ..core.audio import AudioCore
from .base import TestResult

SOUND_PRESETS = [
    {"key": "beep_1k", "name": "1 kHz Beep", "type": "sine", "freq": 1000},
    {"key": "beep_2k", "name": "2 kHz Beep", "type": "sine", "freq": 2000},
    {"key": "beep_5k", "name": "5 kHz Beep", "type": "sine", "freq": 5000},
    {"key": "beep_200", "name": "200 Hz Beep", "type": "sine", "freq": 200},
    {"key": "impulse", "name": "Click (Impulse)", "type": "impulse", "freq": None},
]

class DelayRunner:
    def __init__(self, cfg, log_fn):
        self.cfg = cfg
        self.log = log_fn

    def _gen(self, core, preset):
        if preset["type"]=="sine":
            return core.generate_sine(preset["freq"], duration=core.duration)
        return core.generate_impulse()

    def _record_and_play(self, core: AudioCore, preset, record_margin_s=1.0):
        rec_dur = float(core.duration) + float(record_margin_s)
        rec_data = {"audio": None}
        def record_worker(): rec_data["audio"] = core.record_audio(rec_dur)
        t = threading.Thread(target=record_worker, daemon=True); t.start()
        time.sleep(0.08)
        core.play_mono(core.test_signal)
        t.join()
        delay_ms, _ = AudioCore.find_delay_ms(rec_data["audio"], core.test_signal, core.sample_rate)
        return delay_ms, rec_data["audio"], core.test_signal

    # Calibration
    def calibrate_preset(self, core, preset, repeats=5):
        delays = []
        self.log(f"[CAL] {preset['name']} x{repeats}")
        for i in range(repeats):
            self._gen(core, preset)
            d, _, _ = self._record_and_play(core, preset)
            if d is None: self.log(f"  ✗ {i+1}/{repeats}")
            else: self.log(f"  ✓ {i+1}/{repeats}: {d:.2f} ms")
            delays.append(d); time.sleep(0.15)
        avg, std = safe_mean(delays), safe_std(delays)
        if avg is not None:
            self.cfg["per_sound_offsets_ms"].setdefault(preset["key"], 0.0)
            self.cfg["per_sound_offsets_ms"][preset["key"]] = float(avg)
            self.log(f"  -> baseline {avg:.2f} ± {std:.2f} ms saved")
        return avg, std

    def calibrate_global(self, core, repeats=10):
        pseudo = {"key":"impulse","name":"System (Impulse)","type":"impulse","freq":None}
        self.log(f"[CAL] GLOBAL via Impulse x{repeats}")
        delays=[]
        for i in range(repeats):
            self._gen(core, pseudo); d,_,_ = self._record_and_play(core, pseudo)
            if d is None: self.log(f"  ✗ {i+1}/{repeats}")
            else: self.log(f"  ✓ {i+1}/{repeats}: {d:.2f} ms")
            delays.append(d); time.sleep(0.15)
        avg, std = safe_mean(delays), safe_std(delays)
        if avg is not None:
            self.cfg["global_system_offset_ms"] = float(avg)
            self.log(f"  -> GLOBAL {avg:.2f} ± {std:.2f} ms saved")
        return avg, std

    # Testing
    def run_test(self, core, preset, repeats=5, save_plot_dir=None):
        per_sound = self.cfg["per_sound_offsets_ms"].get(preset["key"], 0.0)
        global_off = float(self.cfg.get("global_system_offset_ms", 0.0))
        calib = per_sound + global_off
        self.log(f"[TEST] {preset['name']} x{repeats}  (calib={calib:.2f} ms)")
        results = []
        first_good = None
        for i in range(repeats):
            self._gen(core, preset)
            d_raw, rec, ref = self._record_and_play(core, preset)
            if d_raw is None:
                self.log(f"  ✗ {i+1}/{repeats}"); results.append(None)
            else:
                d_cal = d_raw - calib; self.log(f"  ✓ {i+1}/{repeats}: {d_cal:.2f} ms")
                results.append(d_cal)
                if first_good is None and rec is not None and ref is not None:
                    first_good = (rec, ref)
            time.sleep(0.12)
        avg, std = safe_mean(results), safe_std(results)
        if avg is not None: self.log(f"  -> avg {avg:.2f} ± {std:.2f} ms")
        if save_plot_dir and first_good and avg is not None:
            rec, ref = first_good
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(save_plot_dir, f"{preset['key']}_plot_{ts}.png")
            self._save_plot(rec, ref, avg, core.sample_rate, out, f"{preset['name']} avg {avg:.1f} ms")
            self.log(f"  saved plot -> {out}")
        return results

    @staticmethod
    def _save_plot(recorded_audio, test_signal, avg_delay_ms, sample_rate, filepath, title):
        plt.figure(figsize=(11,7))
        plt.suptitle(title)
        plt.subplot(3,1,1)
        t_ref = np.linspace(0, len(test_signal)/sample_rate, len(test_signal), endpoint=False)
        plt.plot(t_ref*1000.0, test_signal); plt.title("Reference"); plt.xlabel("ms"); plt.ylabel("amp")
        plt.subplot(3,1,2)
        t_rec = np.linspace(0, len(recorded_audio)/sample_rate, len(recorded_audio), endpoint=False)
        plt.plot(t_rec*1000.0, recorded_audio); plt.title("Recorded"); plt.xlabel("ms"); plt.ylabel("amp")
        plt.subplot(3,1,3)
        correlation = signal.correlate(recorded_audio, test_signal, mode="full")
        lags = np.arange(-len(test_signal)+1, len(recorded_audio))
        corr_time = lags / sample_rate * 1000.0
        plt.plot(corr_time, correlation); plt.axvline(x=avg_delay_ms, linestyle="--", label=f"Avg: {avg_delay_ms:.1f} ms"); plt.legend()
        plt.xlabel("Delay (ms)"); plt.ylabel("corr")
        plt.tight_layout(); plt.savefig(filepath); plt.close()
