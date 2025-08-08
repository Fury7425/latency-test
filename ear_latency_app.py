
import os
import json
import time
import threading
import datetime

import numpy as np
from scipy import signal
import pyaudio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import customtkinter as ctk
from tkinter import filedialog, messagebox

APP_TITLE = "Latency Lab — Sleek + Lab"
CALIBRATION_JSON = "calibration_per_sound.json"

# -------------------------------
# Presets
# -------------------------------
SOUND_PRESETS = [
    {"key": "beep_1k", "name": "1 kHz Beep", "type": "sine", "freq": 1000},
    {"key": "beep_2k", "name": "2 kHz Beep", "type": "sine", "freq": 2000},
    {"key": "beep_5k", "name": "5 kHz Beep", "type": "sine", "freq": 5000},
    {"key": "beep_200", "name": "200 Hz Beep", "type": "sine", "freq": 200},
    {"key": "impulse", "name": "Click (Impulse)", "type": "impulse", "freq": None},
]

# -------------------------------
# Config
# -------------------------------
def default_config():
    return {
        "global_system_offset_ms": 0.0,
        "per_sound_offsets_ms": {p["key"]: 0.0 for p in SOUND_PRESETS},
        "last_settings": {
            "sample_rate": 44100,
            "duration": 0.5,
            "repeats": 5,
            "input_device_index": None,
            "output_device_index": None,
            "output_dir": os.getcwd(),
        },
        "ui": {
            "appearance_mode": "Dark",     # 'Light' | 'Dark' | 'System'
            "color_theme": "blue",         # 'blue' | 'dark-blue' | 'green'
        }
    }

def load_config(path=CALIBRATION_JSON):
    cfg = default_config()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # deep merge with defaults
            for k, v in data.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
            for p in SOUND_PRESETS:
                cfg["per_sound_offsets_ms"].setdefault(p["key"], 0.0)
        except Exception:
            pass
    return cfg

def save_config(cfg, path=CALIBRATION_JSON):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def safe_mean(a):
    vals = [x for x in a if x is not None]
    return float(np.mean(vals)) if vals else None

def safe_std(a):
    vals = [x for x in a if x is not None]
    return float(np.std(vals)) if vals else None

def rms(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x**2)) if x.size > 0 else 0.0

def dbfs(x):
    r = rms(x)
    return 20*np.log10(max(r, 1e-12))

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

# -------------------------------
# Audio Core
# -------------------------------
class AudioCore:
    def __init__(self, sample_rate=44100, chunk_size=1024, duration=0.5, output_device_index=None, input_device_index=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.duration = duration
        self.audio = pyaudio.PyAudio()
        self.stream_output = None
        self.stream_input = None
        self.test_signal = None
        self.recorded_audio = None
        self.output_device_index = output_device_index
        self.input_device_index = input_device_index

    def list_devices(self):
        devs = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            devs.append(info)
        return devs

    def set_devices(self, output_index=None, input_index=None):
        self.output_device_index = output_index
        self.input_device_index = input_index

    def generate_sine(self, freq=1000.0, duration=None, amp=0.8):
        if duration is None:
            duration = self.duration
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        fade_duration = 0.01
        fade_samples = int(fade_duration * self.sample_rate)
        envelope = np.ones_like(t)
        fade_samples = min(fade_samples, max(1, len(t)//2 - 1))
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 1, fade_samples)[::-1]
            envelope[:fade_samples] = fade_in
            envelope[len(t)-fade_samples:] = fade_out
        sig = np.sin(2 * np.pi * freq * t) * envelope * float(amp)
        self.test_signal = sig.astype(np.float32)
        return self.test_signal

    def generate_impulse(self, buffer_duration=0.1, impulse_duration_samples=1, amplitude=0.85):
        num_samples = int(self.sample_rate * buffer_duration)
        sig = np.zeros(num_samples, dtype=np.float32)
        start = max(0, num_samples // 2 - impulse_duration_samples // 2)
        end = min(num_samples, start + impulse_duration_samples)
        sig[start:end] = amplitude
        self.test_signal = sig
        return self.test_signal

    def generate_pink_noise(self, duration=None, amp=0.3):
        if duration is None:
            duration = self.duration
        n = int(self.sample_rate * duration)
        white = np.random.normal(0, 1, n)
        freqs = np.fft.rfftfreq(n, 1/self.sample_rate)
        mag = 1/np.maximum(freqs, 1e-6)
        spectrum = np.fft.rfft(white) * mag
        pink = np.fft.irfft(spectrum, n)
        pink = pink / (np.max(np.abs(pink)) + 1e-9) * amp
        self.test_signal = pink.astype(np.float32)
        return self.test_signal

    def generate_log_chirp(self, f0=20, f1=20000, duration=6.0, amp=0.5):
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        sig = signal.chirp(t, f0=f0, t1=duration, f1=f1, method='logarithmic') * amp
        fade = int(0.02 * self.sample_rate)
        if fade > 0:
            sig[:fade] *= np.linspace(0, 1, fade)
            sig[-fade:] *= np.linspace(1, 0, fade)
        self.test_signal = sig.astype(np.float32)
        return self.test_signal, t

    # Playback
    def play_mono(self, sig):
        audio_data = (np.asarray(sig) * 32767).astype(np.int16)
        out = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True, output_device_index=self.output_device_index)
        out.write(audio_data.tobytes())
        out.stop_stream(); out.close()

    def play_stereo(self, left, right):
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        n = min(len(left), len(right))
        interleaved = np.empty(2*n, dtype=np.float32)
        interleaved[0::2] = left[:n]
        interleaved[1::2] = right[:n]
        audio_data = (interleaved * 32767).astype(np.int16).tobytes()
        out = self.audio.open(format=pyaudio.paInt16, channels=2, rate=self.sample_rate, output=True, output_device_index=self.output_device_index)
        out.write(audio_data)
        out.stop_stream(); out.close()

    # Record
    def record_audio(self, duration):
        inp = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size, input_device_index=self.input_device_index)
        frames = []
        num_chunks = int(self.sample_rate * duration / self.chunk_size)
        for _ in range(num_chunks):
            data = inp.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
        inp.stop_stream(); inp.close()
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32767.0
        self.recorded_audio = audio_data
        return self.recorded_audio

    @staticmethod
    def find_delay_ms(recorded_signal, reference_signal, sample_rate):
        if recorded_signal is None or reference_signal is None:
            return None, None
        if len(recorded_signal) == 0 or np.all(recorded_signal == 0):
            return None, None
        if len(reference_signal) == 0 or np.all(reference_signal == 0):
            return None, None

        rs = recorded_signal / (np.max(np.abs(recorded_signal)) + 1e-9)
        rf = reference_signal / (np.max(np.abs(reference_signal)) + 1e-9)
        correlation = signal.correlate(rs, rf, mode="full")

        peak_index = int(np.argmax(np.abs(correlation)))
        if 0 < peak_index < len(correlation) - 1:
            y1 = correlation[peak_index - 1]
            y2 = correlation[peak_index]
            y3 = correlation[peak_index + 1]
            denom = (y1 - 2*y2 + y3)
            if abs(denom) > 1e-9:
                frac = (y1 - y3) / (2 * denom)
                peak_index_f = peak_index + float(frac)
            else:
                peak_index_f = float(peak_index)
        else:
            peak_index_f = float(peak_index)

        delay_samples = peak_index_f - (len(reference_signal) - 1)
        delay_ms = (delay_samples / sample_rate) * 1000.0
        return delay_ms, correlation

    def cleanup(self):
        try:
            if self.audio:
                self.audio.terminate()
                self.audio = None
        except Exception:
            pass

# -------------------------------
# Delay Runner (measure/calibrate)
# -------------------------------
class DelayRunner:
    def __init__(self, cfg, log_fn):
        self.cfg = cfg
        self.log = log_fn

    def _gen(self, core, preset):
        return core.generate_sine(preset["freq"]) if preset["type"] == "sine" else core.generate_impulse()

    def run_once(self, core, preset, record_margin_s=1.0):
        rec_dur = float(core.duration) + float(record_margin_s)
        rec_data = {"audio": None}

        def record_worker():
            rec_data["audio"] = core.record_audio(rec_dur)

        t = threading.Thread(target=record_worker, daemon=True)
        t.start()
        time.sleep(0.08)  # ensure recording started
        core.play_mono(core.test_signal)
        t.join()

        delay_ms, _ = AudioCore.find_delay_ms(rec_data["audio"], core.test_signal, core.sample_rate)
        return delay_ms, rec_data["audio"], core.test_signal

    def calibrate_preset(self, core, preset, repeats=5):
        self.log(f"[CAL] {preset['name']} x{repeats}")
        delays = []
        for i in range(repeats):
            self._gen(core, preset)
            d, _, _ = self.run_once(core, preset)
            if d is None:
                self.log(f"  ✗ {i+1}/{repeats}: failed")
            else:
                self.log(f"  ✓ {i+1}/{repeats}: {d:.2f} ms")
            delays.append(d)
            time.sleep(0.2)
        avg, std = safe_mean(delays), safe_std(delays)
        if avg is not None:
            self.cfg["per_sound_offsets_ms"][preset["key"]] = float(avg)
            save_config(self.cfg)
            self.log(f"  -> saved baseline {avg:.2f} ± {std:.2f} ms")
        else:
            self.log("  -> no successful runs")
        return avg, std

    def calibrate_global(self, core, repeats=10):
        pseudo = {"key": "impulse", "name": "System (Impulse)", "type": "impulse", "freq": None}
        self.log(f"[CAL] GLOBAL (Impulse) x{repeats}")
        delays = []
        for i in range(repeats):
            self._gen(core, pseudo)
            d, _, _ = self.run_once(core, pseudo)
            if d is None:
                self.log(f"  ✗ {i+1}/{repeats}: failed")
            else:
                self.log(f"  ✓ {i+1}/{repeats}: {d:.2f} ms")
            delays.append(d)
            time.sleep(0.2)
        avg, std = safe_mean(delays), safe_std(delays)
        if avg is not None:
            self.cfg["global_system_offset_ms"] = float(avg)
            save_config(self.cfg)
            self.log(f"  -> saved GLOBAL {avg:.2f} ± {std:.2f} ms")
        else:
            self.log("  -> no successful runs")
        return avg, std

    def test_preset(self, core, preset, repeats=5, save_plot_dir=None):
        self.log(f"[TEST] {preset['name']} x{repeats}")
        per_sound = self.cfg["per_sound_offsets_ms"].get(preset["key"], 0.0)
        global_off = float(self.cfg.get("global_system_offset_ms", 0.0))
        calib_total = per_sound + global_off
        self.log(f"  using calib: per-sound {per_sound:.2f} + global {global_off:.2f} = {calib_total:.2f} ms")

        results = []
        first_good = None
        for i in range(repeats):
            self._gen(core, preset)
            d_raw, rec, ref = self.run_once(core, preset)
            if d_raw is None:
                self.log(f"  ✗ {i+1}/{repeats}: failed")
                results.append(None)
            else:
                d_cal = d_raw - calib_total
                self.log(f"  ✓ {i+1}/{repeats}: raw {d_raw:.2f} -> cal {d_cal:.2f} ms")
                results.append(d_cal)
                if first_good is None and rec is not None and ref is not None:
                    first_good = (rec, ref)
            time.sleep(0.15)

        avg, std = safe_mean(results), safe_std(results)
        if avg is not None:
            self.log(f"  -> avg {avg:.2f} ± {std:.2f} ms")
        else:
            self.log("  -> no successful runs")

        if save_plot_dir and first_good is not None and avg is not None:
            rec, ref = first_good
            try:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(save_plot_dir, f"{preset['key']}_plot_{ts}.png")
                self._save_plot(rec, ref, avg, core.sample_rate, out_path, f"{preset['name']} (avg {avg:.1f} ms)")
                self.log(f"  saved plot: {out_path}")
            except Exception as e:
                self.log(f"  (plot failed: {e})")
        return results

    @staticmethod
    def _save_plot(recorded_audio, test_signal, avg_delay_ms, sample_rate, filepath, title):
        plt.figure(figsize=(11, 7))
        plt.suptitle(title, fontsize=14)

        plt.subplot(3, 1, 1)
        t_ref = np.linspace(0, len(test_signal) / sample_rate, len(test_signal), endpoint=False)
        plt.plot(t_ref * 1000.0, test_signal)
        plt.title('Reference')
        plt.xlabel('Time (ms)'); plt.ylabel('Amplitude')

        plt.subplot(3, 1, 2)
        t_rec = np.linspace(0, len(recorded_audio) / sample_rate, len(recorded_audio), endpoint=False)
        plt.plot(t_rec * 1000.0, recorded_audio)
        plt.title('Recorded')
        plt.xlabel('Time (ms)'); plt.ylabel('Amplitude')

        plt.subplot(3, 1, 3)
        correlation = signal.correlate(recorded_audio, test_signal, mode='full')
        lags = np.arange(-len(test_signal) + 1, len(recorded_audio))
        corr_time = lags / sample_rate * 1000.0
        plt.plot(corr_time, correlation)
        plt.axvline(x=avg_delay_ms, linestyle='--', label=f'Avg Calibrated: {avg_delay_ms:.1f} ms')
        plt.legend(); plt.title('Cross-correlation')
        plt.xlabel('Delay (ms)'); plt.ylabel('Correlation')
        plt.tight_layout(); plt.savefig(filepath); plt.close()

    @staticmethod
    def save_bar(all_delays_by_preset, filepath):
        names, means, stds = [], [], []
        for name, delays in all_delays_by_preset.items():
            d = [x for x in delays if x is not None]
            if not d: 
                continue
            names.append(name)
            means.append(np.mean(d))
            stds.append(np.std(d))
        if not names:
            return False
        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, means, yerr=stds, capsize=5)
        for b in bars:
            y = b.get_height()
            plt.text(b.get_x() + b.get_width()/2, y + 1, f"{y:.1f}", ha='center', va='bottom')
        plt.ylabel("Avg Calibrated Delay (ms)")
        plt.title("Average Calibrated Delay per Sound")
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout(); plt.savefig(filepath); plt.close()
        return True

    @staticmethod
    def save_report(all_results, cfg, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("="*64 + "\\n")
            f.write("LATENCY LAB REPORT\\n")
            f.write("="*64 + "\\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Calibration: GLOBAL {cfg.get('global_system_offset_ms',0.0):.2f} ms + per-sound\\n\\n")

            overall = []
            for name, data in all_results.items():
                f.write(f"\\n--- {name} ---\\n")
                f.write(f"Per-sound baseline: {data['per_sound_baseline_ms']:.2f} ms\\n")
                delays = data["delays_ms"]
                good = [x for x in delays if x is not None]
                if good:
                    f.write("Delays (ms): " + ", ".join(f"{x:.2f}" for x in good) + "\\n")
                    f.write(f"Avg: {np.mean(good):.2f} | Std: {np.std(good):.2f} | N: {len(good)}\\n")
                    overall.extend(good)
                else:
                    f.write("No successful runs.\\n")

            if overall:
                f.write("\\n=== OVERALL ===\\n")
                f.write(f"Overall Avg: {np.mean(overall):.2f} | Std: {np.std(overall):.2f} | N: {len(overall)}\\n")
            f.write("\\nEND\\n")

# -------------------------------
# Lab Tests
# -------------------------------
class LabTests:
    def __init__(self, core, log_fn):
        self.core = core
        self.log = log_fn
        self.results = []

    def test_balance(self, freq=1000.0, tone_dur=1.0, settle=0.2):
        self.log("[BALANCE] Place mic on LEFT earcup, then click OK")
        messagebox.showinfo("Balance", "Place mic on LEFT earcup, then click OK")
        sig = self.core.generate_sine(freq=freq, duration=tone_dur)
        recL = self._play_and_record(sig, both_channels=True, settle=settle)

        self.log("[BALANCE] Move mic to RIGHT earcup, then click OK")
        messagebox.showinfo("Balance", "Move mic to RIGHT earcup, then click OK")
        self.core.generate_sine(freq=freq, duration=tone_dur)
        recR = self._play_and_record(self.core.test_signal, both_channels=True, settle=settle)

        levelL = dbfs(recL); levelR = dbfs(recR)
        diff = levelL - levelR
        res = {
            "test": "balance",
            "timestamp": self._now(),
            "params": {"freq": freq, "duration": tone_dur},
            "metrics": {"left_dBFS": levelL, "right_dBFS": levelR, "L_minus_R_dB": diff}
        }
        self.results.append(res)
        self.log(f"[BALANCE] L {levelL:.2f} dBFS, R {levelR:.2f} dBFS, Δ {diff:.2f} dB")
        return res

    def test_crosstalk(self, freq=1000.0, tone_dur=1.0, settle=0.2, direction="LtoR"):
        if direction == "LtoR":
            self.log("[CROSSTALK] Place mic on LEFT earcup (PRIMARY), click OK")
            messagebox.showinfo("Crosstalk", "Place mic on LEFT earcup (PRIMARY), then click OK")
            sig = self.core.generate_sine(freq=freq, duration=tone_dur)
            rec_primary = self._play_and_record(sig, left_only=True, settle=settle)

            self.log("[CROSSTALK] Move mic to RIGHT earcup (LEAK), click OK")
            messagebox.showinfo("Crosstalk", "Move mic to RIGHT earcup (LEAK), then click OK")
            self.core.generate_sine(freq=freq, duration=tone_dur)
            rec_leak = self._play_and_record(self.core.test_signal, left_only=True, settle=settle)
        else:
            self.log("[CROSSTALK] Place mic on RIGHT earcup (PRIMARY), click OK")
            messagebox.showinfo("Crosstalk", "Place mic on RIGHT earcup (PRIMARY), then click OK")
            sig = self.core.generate_sine(freq=freq, duration=tone_dur)
            rec_primary = self._play_and_record(sig, right_only=True, settle=settle)

            self.log("[CROSSTALK] Move mic to LEFT earcup (LEAK), click OK")
            messagebox.showinfo("Crosstalk", "Move mic to LEFT earcup (LEAK), then click OK")
            self.core.generate_sine(freq=freq, duration=tone_dur)
            rec_leak = self._play_and_record(self.core.test_signal, right_only=True, settle=settle)

        p = rms(rec_primary); l = rms(rec_leak)
        crosstalk_db = 20*np.log10(max(l,1e-12)/max(p,1e-12))
        res = {
            "test": "crosstalk",
            "timestamp": self._now(),
            "params": {"freq": freq, "duration": tone_dur, "direction": direction},
            "metrics": {"primary_rms": p, "leak_rms": l, "crosstalk_dB": crosstalk_db}
        }
        self.results.append(res)
        self.log(f"[CROSSTALK {direction}] {crosstalk_db:.2f} dB (leak vs primary)")
        return res

    def test_sweep_fr(self, f0=20, f1=20000, duration=6.0, save_plot_dir=None):
        self.log("[SWEEP FR] Keep mic steady near driver. Running...")
        sig, t = self.core.generate_log_chirp(f0=f0, f1=f1, duration=duration, amp=0.5)
        rec = self._play_and_record(sig, both_channels=True, settle=0.05, rec_dur=duration+0.5)

        delay_ms, _ = AudioCore.find_delay_ms(rec, sig, self.core.sample_rate)
        delay_s = 0.0 if delay_ms is None else delay_ms/1000.0
        shift = int(round(delay_s * self.core.sample_rate))
        rec_aligned = rec[shift: shift + len(sig)] if shift >=0 else rec[:len(sig)]
        if len(rec_aligned) < len(sig):
            rec_aligned = np.pad(rec_aligned, (0, len(sig)-len(rec_aligned)))

        env = np.abs(signal.hilbert(rec_aligned))
        env = env / (np.max(env)+1e-12)

        T = duration
        freqs_inst = f0 * (f1/f0) ** (t / T)

        grid = np.logspace(np.log10(max(20,f0)), np.log10(min(f1,20000)), 200)
        mag = np.zeros_like(grid)
        for i, f in enumerate(grid):
            idx = np.argmin(np.abs(freqs_inst - f))
            mag[i] = env[idx]

        mag_db = 20*np.log10(np.maximum(mag, 1e-6))
        res = {
            "test": "sweep_fr",
            "timestamp": self._now(),
            "params": {"f0": f0, "f1": f1, "duration": duration},
            "data": {"freqs": grid.tolist(), "mag_db": mag_db.tolist()},
            "metrics": {"delay_ms": delay_ms}
        }
        self.results.append(res)
        self.log(f"[SWEEP FR] Done (estimated delay {delay_ms:.1f} ms)")

        if save_plot_dir:
            ensure_dir(save_plot_dir)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(save_plot_dir, f"sweep_fr_{ts}.png")
            plt.figure(figsize=(9,5))
            plt.semilogx(grid, mag_db)
            plt.xlabel("Frequency (Hz)"); plt.ylabel("Relative Level (dB)")
            plt.title("Relative Frequency Response (not absolute)")
            plt.grid(True, which='both', ls=':', alpha=0.6)
            plt.tight_layout(); plt.savefig(out); plt.close()
            self.log(f"[SWEEP FR] Plot saved -> {out}")
            res.setdefault("files", {})["plot"] = out

        return res

    def test_thd(self, tones=(100, 1000, 6000), tone_dur=1.0, amp=0.6):
        self.log("[THD] Tones: " + ", ".join(str(x) for x in tones))
        thd_list = []
        for f in tones:
            sig = self.core.generate_sine(freq=f, duration=tone_dur, amp=amp)
            rec = self._play_and_record(sig, both_channels=True, settle=0.05)
            thd = self._compute_thd(rec, f)
            thd_list.append({"freq": f, "thd_percent": thd})
            self.log(f"  {f} Hz -> THD {thd:.3f}%")

        res = {
            "test": "thd",
            "timestamp": self._now(),
            "params": {"tones": list(tones), "tone_dur": tone_dur},
            "metrics": {"items": thd_list}
        }
        self.results.append(res)
        return res

    def test_isolation(self, noise_dur=2.0, amp=0.4):
        self.log("[ISOLATION] Place mic near earcup (inside). Click OK.")
        messagebox.showinfo("Isolation", "Place mic near earcup seal (inside). Click OK.")
        sig = self.core.generate_pink_noise(duration=noise_dur, amp=amp)
        rec_in = self._play_and_record(sig, both_channels=True, settle=0.05)

        self.log("[ISOLATION] Move mic 5–10 cm outside the cup. Click OK.")
        messagebox.showinfo("Isolation", "Move mic ~5–10 cm outside the cup. Click OK.")
        self.core.generate_pink_noise(duration=noise_dur, amp=amp)
        rec_out = self._play_and_record(self.core.test_signal, both_channels=True, settle=0.05)

        in_db = dbfs(rec_in); out_db = dbfs(rec_out)
        delta = in_db - out_db
        res = {
            "test": "isolation_inside_out",
            "timestamp": self._now(),
            "params": {"noise_dur": noise_dur},
            "metrics": {"inside_dBFS": in_db, "outside_dBFS": out_db, "delta_dB": delta}
        }
        self.results.append(res)
        self.log(f"[ISOLATION] Inside {in_db:.2f} dBFS, Outside {out_db:.2f} dBFS, Δ {delta:.2f} dB")
        return res

    # helpers
    def _play_and_record(self, mono_signal, left_only=False, right_only=False, both_channels=False, settle=0.05, rec_dur=None):
        if rec_dur is None:
            rec_dur = float(self.core.duration) + 0.3
        rec = {"audio": None}
        def record_worker():
            rec["audio"] = self.core.record_audio(rec_dur)
        t = threading.Thread(target=record_worker, daemon=True)
        t.start()
        time.sleep(0.05 + settle)
        if left_only:
            self.core.play_stereo(mono_signal, np.zeros_like(mono_signal))
        elif right_only:
            self.core.play_stereo(np.zeros_like(mono_signal), mono_signal)
        else:
            if both_channels:
                self.core.play_stereo(mono_signal, mono_signal)
            else:
                self.core.play_mono(mono_signal)
        t.join()
        return rec["audio"]

    def _compute_thd(self, x, f0, n_harm=10):
        sr = self.core.sample_rate
        N = int(2**np.ceil(np.log2(len(x))))
        win = signal.windows.hann(len(x), sym=False)
        X = np.fft.rfft(x*win, n=N)
        freqs = np.fft.rfftfreq(N, 1/sr)
        def bin_mag(f):
            k = np.argmin(np.abs(freqs - f))
            return np.abs(X[k])
        fund = bin_mag(f0)
        harm_power = 0.0
        for k in range(2, n_harm+1):
            fk = f0*k
            if fk > sr/2: break
            harm_power += bin_mag(fk)**2
        thd = (np.sqrt(harm_power) / (np.abs(fund) + 1e-12)) * 100.0
        return float(thd)

    def _now(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def export_last(self, out_dir):
        if not self.results:
            raise RuntimeError("No lab test results yet.")
        ensure_dir(out_dir)
        last = self.results[-1]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(out_dir, f"lab_{last['test']}_{ts}")
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(last, f, indent=2, ensure_ascii=False)
        self._maybe_write_csv(last, base + ".csv")
        return base + ".json"

    def export_all(self, out_dir):
        if not self.results:
            raise RuntimeError("No lab test results yet.")
        ensure_dir(out_dir)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(out_dir, f"lab_ALL_{ts}")
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        csv_path = base + ".csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("test,timestamp,key,value\n")
            for r in self.results:
                if "metrics" in r:
                    for k, v in r["metrics"].items():
                        if isinstance(v, (int,float)):
                            f.write(f"{r['test']},{r['timestamp']},{k},{v}\n")
                if r.get("test") == "thd":
                    for item in r["metrics"].get("items", []):
                        f.write(f"thd,{r['timestamp']},thd_percent_{item['freq']}Hz,{item['thd_percent']}\n")
        return base + ".json"

    def _maybe_write_csv(self, obj, csv_path):
        try:
            if obj.get("test") == "sweep_fr":
                freqs = obj.get("data", {}).get("freqs", [])
                mags = obj.get("data", {}).get("mag_db", [])
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("freq,mag_db\n")
                    for fr, mg in zip(freqs, mags):
                        f.write(f"{fr},{mg}\n")
        except Exception:
            pass

# -------------------------------
# UI
# -------------------------------
class SleekApp(ctk.CTk):
    def __init__(self):
        self.cfg = load_config()
        # Apply UI theme
        ctk.set_appearance_mode(self.cfg["ui"].get("appearance_mode", "Dark"))
        ctk.set_default_color_theme(self.cfg["ui"].get("color_theme", "blue"))
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x780")
        self.minsize(1024, 700)

        self.core = AudioCore(
            sample_rate=self.cfg["last_settings"]["sample_rate"],
            chunk_size=1024,
            duration=self.cfg["last_settings"]["duration"],
            output_device_index=self.cfg["last_settings"]["output_device_index"],
            input_device_index=self.cfg["last_settings"]["input_device_index"],
        )
        self.runner = DelayRunner(self.cfg, self._log)
        self.lab = LabTests(self.core, self._log)

        self._build_layout()
        self._refresh_devices()

        self.worker = None
        self._busy(False)

    def _build_layout(self):
        # Sidebar
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, corner_radius=0, width=240)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_rowconfigure(15, weight=1)

        self.logo = ctk.CTkLabel(self.sidebar, text="Latency Lab", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo.grid(row=0, column=0, padx=18, pady=(18, 8), sticky="w")

        self.status_chip = ctk.CTkLabel(self.sidebar, text="IDLE", fg_color="#2d3748", corner_radius=6, padx=8, pady=4)
        self.status_chip.grid(row=1, column=0, padx=18, pady=(0, 12), sticky="w")

        # Nav
        btn = lambda text, key: ctk.CTkButton(self.sidebar, text=text, command=lambda: self._show_page(key))
        self.btn_measure = btn("Measure", "measure")
        self.btn_calib = btn("Calibrate", "calib")
        self.btn_lab = btn("Lab Tests", "lab")
        self.btn_devices = btn("Devices / Settings", "devices")
        self.btn_results = btn("Results / Export", "results")

        self.btn_measure.grid(row=2, column=0, padx=16, pady=6, sticky="ew")
        self.btn_calib.grid(row=3, column=0, padx=16, pady=6, sticky="ew")
        self.btn_lab.grid(row=4, column=0, padx=16, pady=6, sticky="ew")
        self.btn_devices.grid(row=5, column=0, padx=16, pady=6, sticky="ew")
        self.btn_results.grid(row=6, column=0, padx=16, pady=6, sticky="ew")

        # Theme controls
        ctk.CTkLabel(self.sidebar, text="Appearance").grid(row=8, column=0, padx=18, pady=(16, 4), sticky="w")
        self.appearance_menu = ctk.CTkOptionMenu(
            self.sidebar, values=["Light", "Dark", "System"],
            command=self._on_change_appearance
        )
        self.appearance_menu.set(self.cfg["ui"].get("appearance_mode", "Dark"))
        self.appearance_menu.grid(row=9, column=0, padx=18, sticky="ew")

        ctk.CTkLabel(self.sidebar, text="Accent Color").grid(row=10, column=0, padx=18, pady=(12, 4), sticky="w")
        self.color_menu = ctk.CTkOptionMenu(
            self.sidebar, values=["blue", "dark-blue", "green"],
            command=self._on_change_color_theme
        )
        self.color_menu.set(self.cfg["ui"].get("color_theme", "blue"))
        self.color_menu.grid(row=11, column=0, padx=18, sticky="ew")

        self.progress = ctk.CTkProgressBar(self.sidebar, mode="indeterminate")
        self.progress.grid(row=12, column=0, padx=16, pady=12, sticky="ew")

        # Main area
        self.main = ctk.CTkFrame(self, corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # Pages
        self.pages = {}
        self.pages["measure"] = self._build_measure_page(self.main)
        self.pages["calib"] = self._build_calib_page(self.main)
        self.pages["lab"] = self._build_lab_page(self.main)
        self.pages["devices"] = self._build_devices_page(self.main)
        self.pages["results"] = self._build_results_page(self.main)
        self._show_page("measure")

    # ---------- Pages
    def _kpi_card(self, parent, title, value, col):
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=0, column=col, padx=6, sticky="ew")
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=13)).pack(anchor="w", padx=12, pady=(10,0))
        val = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=24, weight="bold"))
        val.pack(anchor="w", padx=12, pady=(4,12))
        return val

    def _build_measure_page(self, parent):
        page = ctk.CTkFrame(parent)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(2, weight=1)

        header = ctk.CTkLabel(page, text="Measure", font=ctk.CTkFont(size=18, weight="bold"))
        header.grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")

        left = ctk.CTkFrame(page)
        left.grid(row=1, column=0, rowspan=2, padx=18, pady=12, sticky="nsw")

        self.preset_vars = {}
        for i, p in enumerate(SOUND_PRESETS):
            var = ctk.BooleanVar(value=(p["key"] != "impulse"))
            self.preset_vars[p["key"]] = var
            ctk.CTkSwitch(left, text=p["name"], variable=var).grid(row=i, column=0, padx=12, pady=6, sticky="w")

        ctk.CTkLabel(left, text="Repeats").grid(row=len(SOUND_PRESETS), column=0, padx=12, pady=(12, 0), sticky="w")
        self.repeats_var = ctk.IntVar(value=self.cfg["last_settings"]["repeats"])
        self.repeats_slider = ctk.CTkSlider(left, from_=1, to=20, number_of_steps=19, command=lambda v: self.repeats_label.configure(text=str(int(v))))
        self.repeats_slider.set(self.repeats_var.get())
        self.repeats_slider.grid(row=len(SOUND_PRESETS)+1, column=0, padx=12, pady=(2, 4), sticky="ew")
        self.repeats_label = ctk.CTkLabel(left, text=str(self.repeats_var.get()))
        self.repeats_label.grid(row=len(SOUND_PRESETS)+2, column=0, padx=12, pady=(0, 12), sticky="w")

        self.save_plots_var = ctk.BooleanVar(value=True)
        self.save_bar_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(left, text="Save per-sound plot", variable=self.save_plots_var).grid(row=len(SOUND_PRESETS)+3, column=0, padx=12, pady=4, sticky="w")
        ctk.CTkSwitch(left, text="Save overall bar chart", variable=self.save_bar_var).grid(row=len(SOUND_PRESETS)+4, column=0, padx=12, pady=4, sticky="w")

        ctk.CTkLabel(left, text="Output Folder").grid(row=len(SOUND_PRESETS)+5, column=0, padx=12, pady=(12, 4), sticky="w")
        self.output_dir_var = ctk.StringVar(value=self.cfg["last_settings"].get("output_dir", os.getcwd()))
        out_row = ctk.CTkFrame(left)
        out_row.grid(row=len(SOUND_PRESETS)+6, column=0, padx=12, pady=(0, 8), sticky="ew")
        out_row.grid_columnconfigure(0, weight=1)
        self.output_entry = ctk.CTkEntry(out_row, textvariable=self.output_dir_var)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0,8))
        ctk.CTkButton(out_row, text="Browse", command=self._choose_outdir, width=80).grid(row=0, column=1)

        btn_row = ctk.CTkFrame(left)
        btn_row.grid(row=len(SOUND_PRESETS)+7, column=0, padx=12, pady=8, sticky="ew")
        ctk.CTkButton(btn_row, text="Run Selected", command=self._on_run_selected).grid(row=0, column=0, padx=(0,6))
        ctk.CTkButton(btn_row, text="Run ALL", command=self._on_run_all).grid(row=0, column=1, padx=6)

        right = ctk.CTkFrame(page)
        right.grid(row=1, column=1, padx=18, pady=12, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        cards = ctk.CTkFrame(right)
        cards.grid(row=0, column=0, sticky="ew", pady=(0,12))
        for i in range(3):
            cards.grid_columnconfigure(i, weight=1)
        self.kpi_avg = self._kpi_card(cards, "Average (ms)", "--", 0)
        self.kpi_std = self._kpi_card(cards, "Std Dev (ms)", "--", 1)
        self.kpi_last = self._kpi_card(cards, "Last (ms)", "--", 2)

        self.summary_box = ctk.CTkTextbox(right, wrap="word")
        self.summary_box.grid(row=1, column=0, sticky="nsew")
        self.summary_box.insert("1.0", "Ready.\\n")

        ctk.CTkButton(right, text="Save Text Report", command=self._on_save_report).grid(row=2, column=0, sticky="e", pady=8)

        return page

    def _build_calib_page(self, parent):
        page = ctk.CTkFrame(parent)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(1, weight=1)

        header = ctk.CTkLabel(page, text="Calibrate", font=ctk.CTkFont(size=18, weight="bold"))
        header.grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")

        left = ctk.CTkFrame(page)
        left.grid(row=1, column=0, padx=18, pady=12, sticky="nsw")

        self.cal_vars = {}
        for i, p in enumerate(SOUND_PRESETS):
            var = ctk.BooleanVar(value=True)
            self.cal_vars[p["key"]] = var
            ctk.CTkSwitch(left, text=p["name"], variable=var).grid(row=i, column=0, padx=12, pady=6, sticky="w")

        ctk.CTkLabel(left, text="Repeats").grid(row=len(SOUND_PRESETS), column=0, padx=12, pady=(12, 0), sticky="w")
        self.cal_repeats_var = ctk.IntVar(value=5)
        slider = ctk.CTkSlider(left, from_=1, to=20, number_of_steps=19, command=lambda v: reps_lab.configure(text=str(int(v))))
        slider.set(self.cal_repeats_var.get())
        slider.grid(row=len(SOUND_PRESETS)+1, column=0, padx=12, pady=(2, 4), sticky="ew")
        reps_lab = ctk.CTkLabel(left, text=str(self.cal_repeats_var.get()))
        reps_lab.grid(row=len(SOUND_PRESETS)+2, column=0, padx=12, pady=(0, 12), sticky="w")

        ctk.CTkButton(left, text="Calibrate Selected Presets", command=self._on_cal_selected).grid(row=len(SOUND_PRESETS)+3, column=0, padx=12, pady=(6,4), sticky="ew")
        ctk.CTkButton(left, text="Calibrate ALL Presets", command=self._on_cal_all).grid(row=len(SOUND_PRESETS)+4, column=0, padx=12, pady=4, sticky="ew")
        ctk.CTkButton(left, text="Calibrate GLOBAL (Impulse)", command=self._on_cal_global).grid(row=len(SOUND_PRESETS)+5, column=0, padx=12, pady=(4,12), sticky="ew")

        right = ctk.CTkFrame(page)
        right.grid(row=1, column=1, padx=18, pady=12, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        self.baseline_box = ctk.CTkTextbox(right, wrap="word")
        self.baseline_box.grid(row=0, column=0, sticky="nsew")
        self._refresh_baseline_box()

        return page

    def _build_lab_page(self, parent):
        page = ctk.CTkFrame(parent)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(2, weight=1)

        header = ctk.CTkLabel(page, text="Lab Tests", font=ctk.CTkFont(size=18, weight="bold"))
        header.grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")

        # Balance
        bal = ctk.CTkFrame(page)
        bal.grid(row=1, column=0, padx=18, pady=12, sticky="nsew")
        ctk.CTkLabel(bal, text="Channel Balance").grid(row=0, column=0, sticky="w", padx=8, pady=(8,4))
        self.var_bal_freq = ctk.StringVar(value="1000")
        ctk.CTkLabel(bal, text="Freq (Hz)").grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkEntry(bal, textvariable=self.var_bal_freq, width=90).grid(row=1, column=1, sticky="w", padx=8)
        ctk.CTkButton(bal, text="Run", command=self._on_lab_balance).grid(row=2, column=0, padx=8, pady=8, sticky="w")

        # Crosstalk
        xt = ctk.CTkFrame(page)
        xt.grid(row=1, column=1, padx=18, pady=12, sticky="nsew")
        ctk.CTkLabel(xt, text="Crosstalk").grid(row=0, column=0, sticky="w", padx=8, pady=(8,4))
        self.var_xt_freq = ctk.StringVar(value="1000")
        ctk.CTkLabel(xt, text="Freq (Hz)").grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkEntry(xt, textvariable=self.var_xt_freq, width=90).grid(row=1, column=1, sticky="w", padx=8)
        self.var_xt_dir = ctk.StringVar(value="LtoR")
        ctk.CTkOptionMenu(xt, values=["LtoR", "RtoL"], variable=self.var_xt_dir).grid(row=1, column=2, padx=8)
        ctk.CTkButton(xt, text="Run", command=self._on_lab_crosstalk).grid(row=2, column=0, padx=8, pady=8, sticky="w")

        # Sweep FR
        fr = ctk.CTkFrame(page)
        fr.grid(row=2, column=0, padx=18, pady=12, sticky="nsew")
        ctk.CTkLabel(fr, text="Sweep FR (relative)").grid(row=0, column=0, sticky="w", padx=8, pady=(8,4))
        self.var_fr_dur = ctk.StringVar(value="6.0")
        ctk.CTkLabel(fr, text="Duration (s)").grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkEntry(fr, textvariable=self.var_fr_dur, width=90).grid(row=1, column=1, sticky="w", padx=8)
        ctk.CTkButton(fr, text="Run", command=self._on_lab_sweep).grid(row=2, column=0, padx=8, pady=8, sticky="w")

        # THD
        thd = ctk.CTkFrame(page)
        thd.grid(row=2, column=1, padx=18, pady=12, sticky="nsew")
        ctk.CTkLabel(thd, text="THD (100/1k/6k Hz)").grid(row=0, column=0, sticky="w", padx=8, pady=(8,4))
        ctk.CTkButton(thd, text="Run", command=self._on_lab_thd).grid(row=1, column=0, padx=8, pady=8, sticky="w")

        # Isolation
        iso = ctk.CTkFrame(page)
        iso.grid(row=3, column=0, padx=18, pady=12, sticky="nsew")
        ctk.CTkLabel(iso, text="Isolation (Inside → Outside)").grid(row=0, column=0, sticky="w", padx=8, pady=(8,4))
        ctk.CTkButton(iso, text="Run", command=self._on_lab_isolation).grid(row=1, column=0, padx=8, pady=8, sticky="w")

        # Results + export
        labres = ctk.CTkFrame(page)
        labres.grid(row=3, column=1, padx=18, pady=12, sticky="nsew")
        labres.grid_rowconfigure(0, weight=1)
        labres.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(labres, text="Lab Results").grid(row=0, column=0, sticky="w", padx=8, pady=(8,4))
        self.txt_lab = ctk.CTkTextbox(labres)
        self.txt_lab.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        row = ctk.CTkFrame(labres); row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0,8))
        ctk.CTkButton(row, text="Export LAST (JSON/CSV)", command=self._on_lab_export_last).pack(side="left", padx=4)
        ctk.CTkButton(row, text="Export ALL (JSON/CSV)", command=self._on_lab_export_all).pack(side="left", padx=4)

        return page

    def _build_devices_page(self, parent):
        page = ctk.CTkFrame(parent)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(1, weight=1)

        header = ctk.CTkLabel(page, text="Devices / Settings", font=ctk.CTkFont(size=18, weight="bold"))
        header.grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")

        # Audio parameters
        card = ctk.CTkFrame(page)
        card.grid(row=1, column=0, columnspan=2, padx=18, pady=12, sticky="ew")
        for i in range(6):
            card.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(card, text="Sample Rate").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.sr_var = ctk.IntVar(value=self.cfg["last_settings"]["sample_rate"])
        ctk.CTkEntry(card, textvariable=self.sr_var, width=110).grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ctk.CTkLabel(card, text="Signal Duration (s)").grid(row=0, column=2, padx=8, pady=8, sticky="w")
        self.dur_var = ctk.DoubleVar(value=self.cfg["last_settings"]["duration"])
        ctk.CTkEntry(card, textvariable=self.dur_var, width=110).grid(row=0, column=3, padx=8, pady=8, sticky="w")

        ctk.CTkButton(card, text="Apply", command=self._on_apply_audio).grid(row=0, column=5, padx=8, pady=8, sticky="e")

        # Device selectors
        dev_card = ctk.CTkFrame(page)
        dev_card.grid(row=2, column=0, columnspan=2, padx=18, pady=8, sticky="ew")
        dev_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(dev_card, text="Output Device").grid(row=0, column=0, padx=8, pady=(8,4), sticky="w")
        self.out_menu = ctk.CTkOptionMenu(dev_card, values=["<refresh>"], command=lambda _: None)
        self.out_menu.grid(row=0, column=1, padx=8, pady=(8,4), sticky="ew")

        ctk.CTkLabel(dev_card, text="Input Device").grid(row=1, column=0, padx=8, pady=(4,8), sticky="w")
        self.in_menu = ctk.CTkOptionMenu(dev_card, values=["<refresh>"], command=lambda _: None)
        self.in_menu.grid(row=1, column=1, padx=8, pady=(4,8), sticky="ew")

        btns = ctk.CTkFrame(page)
        btns.grid(row=3, column=0, columnspan=2, padx=18, pady=8, sticky="ew")
        btns.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(btns, text="Refresh Devices", command=self._refresh_devices).grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ctk.CTkButton(btns, text="Use Selected Devices", command=self._on_use_devices).grid(row=0, column=1, padx=6, pady=6, sticky="e")

        return page

    def _build_results_page(self, parent):
        page = ctk.CTkFrame(parent)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_rowconfigure(0, weight=1)
        page.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(page, text="Results & Export", font=ctk.CTkFont(size=18, weight="bold"))
        header.grid(row=0, column=0, padx=18, pady=(18, 4), sticky="w")

        self.log_box = ctk.CTkTextbox(page, wrap="word")
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=18, pady=12)

        btns = ctk.CTkFrame(page)
        btns.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        btns.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(btns, text="Copy Log", command=self._copy_log, width=110).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(btns, text="Clear Log", command=lambda: self.log_box.delete("1.0", "end"), width=110).grid(row=0, column=1, padx=6, sticky="w")

        return page

    # ---------- Nav & Theme
    def _show_page(self, key):
        for k, p in self.pages.items():
            p.grid_remove()
        self.pages[key].grid()
        self.status_chip.configure(text=key.upper())

    def _on_change_appearance(self, value):
        try:
            ctk.set_appearance_mode(value)
            self.cfg["ui"]["appearance_mode"] = value
            save_config(self.cfg)
        except Exception as e:
            messagebox.showerror("Appearance", str(e))

    def _on_change_color_theme(self, value):
        try:
            ctk.set_default_color_theme(value)
            self.cfg["ui"]["color_theme"] = value
            save_config(self.cfg)
        except Exception as e:
            messagebox.showerror("Color Theme", str(e))

    # ---------- Handlers (Measure/Calib)
    def _choose_outdir(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir_var.set(d)
            self.cfg["last_settings"]["output_dir"] = d
            save_config(self.cfg)

    def _on_run_selected(self):
        presets = [p for p in SOUND_PRESETS if self.preset_vars[p["key"]].get()]
        if not presets:
            messagebox.showinfo("Measure", "Select at least one preset.")
            return
        self._start(self._run_tests, presets)

    def _on_run_all(self):
        self._start(self._run_tests, SOUND_PRESETS[:])

    def _on_save_report(self):
        if not hasattr(self, "last_all_results"):
            messagebox.showinfo("Report", "Run tests first.")
            return
        out_dir = self.output_dir_var.get() or os.getcwd()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rep_path = os.path.join(out_dir, f"latency_report_{ts}.txt")
        DelayRunner.save_report(self.last_all_results, self.cfg, rep_path)
        self._log(f"[SAVE] report -> {rep_path}")
        messagebox.showinfo("Report", f"Saved:\\n{rep_path}")

    def _on_cal_selected(self):
        presets = [p for p in SOUND_PRESETS if self.cal_vars[p["key"]].get()]
        if not presets:
            messagebox.showinfo("Calibrate", "Select at least one preset.")
            return
        self._start(self._cal_presets, presets)

    def _on_cal_all(self):
        self._start(self._cal_presets, SOUND_PRESETS[:])

    def _on_cal_global(self):
        self._start(self._cal_global)

    def _on_apply_audio(self):
        try:
            self.core.sample_rate = int(self.sr_var.get())
            self.core.duration = float(self.dur_var.get())
            self.cfg["last_settings"]["sample_rate"] = self.core.sample_rate
            self.cfg["last_settings"]["duration"] = self.core.duration
            save_config(self.cfg)
            self._log(f"[SET] sr={self.core.sample_rate}, dur={self.core.duration}s")
        except Exception as e:
            messagebox.showerror("Settings", str(e))

    def _on_use_devices(self):
        out_idx = self._out_indices.get(self.out_menu.get())
        in_idx = self._in_indices.get(self.in_menu.get())
        if out_idx is None or in_idx is None:
            messagebox.showwarning("Devices", "Select both output and input devices.")
            return
        self.core.set_devices(output_index=out_idx, input_index=in_idx)
        self.cfg["last_settings"]["output_device_index"] = out_idx
        self.cfg["last_settings"]["input_device_index"] = in_idx
        save_config(self.cfg)
        self._log(f"[SET] output={out_idx}, input={in_idx}")

    # ---------- Handlers (Lab)
    def _on_lab_balance(self):
        try:
            freq = float(self.var_bal_freq.get())
            res = self.lab.test_balance(freq=freq)
            self._append_lab(res)
        except Exception as e:
            messagebox.showerror("Balance", str(e))

    def _on_lab_crosstalk(self):
        try:
            freq = float(self.var_xt_freq.get())
            direction = self.var_xt_dir.get()
            res = self.lab.test_crosstalk(freq=freq, direction=direction)
            self._append_lab(res)
        except Exception as e:
            messagebox.showerror("Crosstalk", str(e))

    def _on_lab_sweep(self):
        try:
            dur = float(self.var_fr_dur.get())
            out_dir = self.output_dir_var.get() or os.getcwd()
            res = self.lab.test_sweep_fr(duration=dur, save_plot_dir=out_dir)
            self._append_lab(res)
        except Exception as e:
            messagebox.showerror("Sweep FR", str(e))

    def _on_lab_thd(self):
        try:
            res = self.lab.test_thd()
            self._append_lab(res)
        except Exception as e:
            messagebox.showerror("THD", str(e))

    def _on_lab_isolation(self):
        try:
            res = self.lab.test_isolation()
            self._append_lab(res)
        except Exception as e:
            messagebox.showerror("Isolation", str(e))

    def _on_lab_export_last(self):
        try:
            out_dir = self.output_dir_var.get() or os.getcwd()
            path = self.lab.export_last(out_dir)
            self._log(f"[EXPORT] Last lab result -> {path}")
            messagebox.showinfo("Export", f"Saved:\\n{path}")
        except Exception as e:
            messagebox.showerror("Export", str(e))

    def _on_lab_export_all(self):
        try:
            out_dir = self.output_dir_var.get() or os.getcwd()
            path = self.lab.export_all(out_dir)
            self._log(f"[EXPORT] All lab results -> {path}")
            messagebox.showinfo("Export", f"Saved:\\n{path}")
        except Exception as e:
            messagebox.showerror("Export", str(e))

    # ---------- Workers
    def _start(self, fn, *args):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Another task is running.")
            return
        self._busy(True)
        self.worker = threading.Thread(target=self._guard, args=(fn, *args), daemon=True)
        self.worker.start()

    def _guard(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            self._log(f"[ERROR] {e}")
        finally:
            self._busy(False)

    def _run_tests(self, presets):
        repeats = int(self.repeats_slider.get())
        out_dir = self.output_dir_var.get() or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        all_results = {}
        all_delays_bar = {}
        summary_lines = ["=== SUMMARY (Calibrated) ==="]

        the_last = None
        overall_vals = []

        for p in presets:
            results = self.runner.test_preset(
                self.core, p, repeats=repeats,
                save_plot_dir=out_dir if self.save_plots_var.get() else None
            )
            delays = [r for r in results if r is not None]
            all_results[p["name"]] = {
                "per_sound_baseline_ms": self.cfg["per_sound_offsets_ms"].get(p["key"], 0.0),
                "delays_ms": delays
            }
            all_delays_bar[p["name"]] = delays

            if delays:
                avg, std = float(np.mean(delays)), float(np.std(delays))
                summary_lines.append(f"{p['name']}: avg {avg:.2f} ms | std {std:.2f} | n {len(delays)}")
                overall_vals.extend(delays)
                the_last = delays[-1]

        self.last_all_results = all_results
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", "\\n".join(summary_lines))

        # KPI
        if overall_vals:
            self.kpi_avg.configure(text=f"{np.mean(overall_vals):.1f}")
            self.kpi_std.configure(text=f"{np.std(overall_vals):.1f}")
            self.kpi_last.configure(text=f"{the_last:.1f}")
        else:
            self.kpi_avg.configure(text="--")
            self.kpi_std.configure(text="--")
            self.kpi_last.configure(text="--")

        # Overall bar chart
        if self.save_bar_var.get():
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bar_path = os.path.join(out_dir, f"overall_bar_{ts}.png")
            ok = DelayRunner.save_bar(all_delays_by_preset=all_delays_bar, filepath=bar_path)
            if ok:
                self._log(f"[SAVE] bar chart -> {bar_path}")

        self._log("[DONE] tests complete.")

    def _cal_presets(self, presets):
        reps = int(self.cal_repeats_var.get())
        for p in presets:
            self.runner.calibrate_preset(self.core, p, repeats=reps)
        self._refresh_baseline_box()
        self._log("[DONE] per-sound calibration complete.")

    def _cal_global(self):
        reps = int(self.cal_repeats_var.get())
        val, _ = self.runner.calibrate_global(self.core, repeats=reps)
        if val is not None:
            self._log(f"[CAL] GLOBAL set to {val:.2f} ms")
        self._refresh_baseline_box()
        self._log("[DONE] global calibration complete.")

    # ---------- Helpers
    def _refresh_baseline_box(self):
        lines = ["Per-sound baselines (ms):"]
        for p in SOUND_PRESETS:
            val = self.cfg["per_sound_offsets_ms"].get(p["key"], 0.0)
            lines.append(f"• {p['name']}: {val:.2f}")
        lines.append(f"\\nGLOBAL offset: {self.cfg.get('global_system_offset_ms', 0.0):.2f} ms")
        if hasattr(self, "baseline_box"):
            self.baseline_box.delete("1.0", "end")
            self.baseline_box.insert("1.0", "\\n".join(lines))

    def _refresh_devices(self):
        devs = self.core.list_devices()
        out_values = []
        in_values = []
        self._out_indices = {}
        self._in_indices = {}

        for i, info in enumerate(devs):
            name = info.get("name", "unknown")
            max_in = int(info.get("maxInputChannels", 0))
            max_out = int(info.get("maxOutputChannels", 0))
            rate = int(info.get("defaultSampleRate", 0))
            label = f"[{i}] {name}  (in:{max_in}, out:{max_out}, rate:{rate})"
            if max_out > 0:
                out_values.append(label)
                self._out_indices[label] = i
            if max_in > 0:
                in_values.append(label)
                self._in_indices[label] = i

        if not out_values: out_values = ["<no output devices>"]
        if not in_values: in_values = ["<no input devices>"]

        self.out_menu.configure(values=out_values)
        self.in_menu.configure(values=in_values)

        # Preselect last-used
        def pick(menu, values, target_idx):
            if target_idx is None or not values:
                menu.set(values[0])
                return
            for v in values:
                if self._out_indices.get(v, None) == target_idx or self._in_indices.get(v, None) == target_idx:
                    menu.set(v); return
            menu.set(values[0])

        pick(self.out_menu, out_values, self.cfg["last_settings"].get("output_device_index"))
        pick(self.in_menu, in_values, self.cfg["last_settings"].get("input_device_index"))

    def _busy(self, is_busy: bool):
        if is_busy:
            self.status_chip.configure(text="WORKING", fg_color="#1f2937")
            self.progress.start()
        else:
            self.status_chip.configure(text="IDLE", fg_color="#2d3748")
            self.progress.stop()

    def _log(self, msg):
        if hasattr(self, "log_box"):
            self.log_box.insert("end", msg + "\\n")
            self.log_box.see("end")

    def _copy_log(self):
        text = self.log_box.get("1.0", "end")
        self.clipboard_clear()
        self.clipboard_append(text)

    def _append_lab(self, res: dict):
        if hasattr(self, "txt_lab"):
            self.txt_lab.insert("end", json.dumps(res, indent=2, ensure_ascii=False) + "\\n\\n")
            self.txt_lab.see("end")

if __name__ == "__main__":
    app = SleekApp()
    app.mainloop()
