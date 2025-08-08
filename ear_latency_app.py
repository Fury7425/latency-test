
import os
import json
import time
import threading
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from scipy import signal, fft

import pyaudio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


APP_TITLE = "Ear/Headset Latency Tester (with Lab)"
CALIBRATION_JSON = "calibration_per_sound.json"


# -------------------------------
# Utility
# -------------------------------
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

    # --- Generators ---
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
        # Pink noise via 1/f shaping in frequency domain
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
        # fade in/out
        fade = int(0.02 * self.sample_rate)
        if fade > 0:
            sig[:fade] *= np.linspace(0, 1, fade)
            sig[-fade:] *= np.linspace(1, 0, fade)
        self.test_signal = sig.astype(np.float32)
        return self.test_signal, t

    # --- Playback ---
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

    # --- Record ---
    def record_audio(self, duration):
        inp = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size, input_device_index=self.input_device_index)
        frames = []
        num_chunks = int(self.sample_rate * duration / self.chunk_size)
        for _ in range(num_chunks):
            data = inp.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
        inp.stop_stream(); inp.close()
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32767.0
        return audio_data

    # --- Analysis ---
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
            self.audio.terminate()
        except Exception:
            pass


# -------------------------------
# Presets & Config
# -------------------------------
SOUND_PRESETS = [
    {"key": "beep_1k", "name": "1 kHz Beep", "type": "sine", "freq": 1000},
    {"key": "beep_5k", "name": "5 kHz Beep", "type": "sine", "freq": 5000},
    {"key": "beep_200", "name": "200 Hz Beep", "type": "sine", "freq": 200},
    {"key": "beep_2k", "name": "2 kHz Beep", "type": "sine", "freq": 2000},
    {"key": "impulse", "name": "Click (Impulse)", "type": "impulse", "freq": None},
]

def default_config():
    return {
        "global_system_offset_ms": 0.0,
        "per_sound_offsets_ms": {p["key"]: 0.0 for p in SOUND_PRESETS},
        "last_settings": {
            "sample_rate": 44100,
            "duration": 0.5,
            "repeats": 5,
            "input_device_index": None,
            "output_device_index": None
        }
    }

def load_config(path=CALIBRATION_JSON):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = default_config()
            cfg.update(data)
            for p in SOUND_PRESETS:
                if p["key"] not in cfg["per_sound_offsets_ms"]:
                    cfg["per_sound_offsets_ms"][p["key"]] = 0.0
            return cfg
        except Exception:
            return default_config()
    return default_config()

def save_config(cfg, path=CALIBRATION_JSON):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


# -------------------------------
# Delay Runner (original tests)
# -------------------------------
class DelayRunner:
    def __init__(self, cfg, log_fn):
        self.cfg = cfg
        self.log = log_fn

    def _gen_signal(self, core: AudioCore, preset):
        if preset["type"] == "sine":
            return core.generate_sine(freq=preset["freq"], duration=core.duration)
        else:
            return core.generate_impulse()

    def run_single_measurement(self, core: AudioCore, preset, record_margin_s=1.0):
        rec_dur = float(core.duration) + float(record_margin_s)
        rec_data = {"audio": None}

        def record_worker():
            rec_data["audio"] = core.record_audio(rec_dur)

        t = threading.Thread(target=record_worker, daemon=True)
        t.start()
        time.sleep(0.1)
        core.play_mono(core.test_signal)
        t.join()

        delay_ms, _ = AudioCore.find_delay_ms(rec_data["audio"], core.test_signal, core.sample_rate)
        return delay_ms, rec_data["audio"], core.test_signal

    def run_calibration_for_preset(self, core: AudioCore, preset, repeats=5):
        self.log(f"[CAL] {preset['name']} - {repeats} runs")
        delays = []
        for i in range(repeats):
            self._gen_signal(core, preset)
            d, _, _ = self.run_single_measurement(core, preset)
            if d is None:
                self.log(f"  ✗ Run {i+1}/{repeats}: failed")
            else:
                self.log(f"  ✓ Run {i+1}/{repeats}: raw {d:.3f} ms")
            delays.append(d)
            time.sleep(0.2)

        avg = safe_mean(delays)
        std = safe_std(delays)
        if avg is None:
            self.log("  -> No successful runs.")
            return None
        self.cfg["per_sound_offsets_ms"][preset["key"]] = float(avg)
        save_config(self.cfg)
        self.log(f"  -> Saved per-sound baseline: {avg:.3f} ms (std {std:.3f} ms)")
        return avg

    def run_global_system_calibration(self, core: AudioCore, repeats=10):
        pseudo_preset = {"key": "impulse", "name": "System (Impulse)", "type": "impulse", "freq": None}
        self.log(f"[CAL] Global offset via impulse - {repeats} runs")
        delays = []
        for i in range(repeats):
            self._gen_signal(core, pseudo_preset)
            d, _, _ = self.run_single_measurement(core, pseudo_preset)
            if d is None:
                self.log(f"  ✗ Run {i+1}/{repeats}: failed")
            else:
                self.log(f"  ✓ Run {i+1}/{repeats}: raw {d:.3f} ms")
            delays.append(d)
            time.sleep(0.2)

        avg = safe_mean(delays)
        std = safe_std(delays)
        if avg is None:
            self.log("  -> No successful global calibration runs.")
            return None
        self.cfg["global_system_offset_ms"] = float(avg)
        save_config(self.cfg)
        self.log(f"  -> Saved GLOBAL system offset: {avg:.3f} ms (std {std:.3f} ms)")
        return avg

    def run_test_for_preset(self, core: AudioCore, preset, repeats=5, save_plot_dir=None):
        self.log(f"[TEST] {preset['name']} - {repeats} runs")
        per_sound = self.cfg["per_sound_offsets_ms"].get(preset["key"], 0.0)
        global_off = float(self.cfg.get("global_system_offset_ms", 0.0))
        calib_total = per_sound + global_off
        self.log(f"  Using calibration: per-sound {per_sound:.3f} ms + global {global_off:.3f} ms = total {calib_total:.3f} ms")

        results = []
        first_good = None
        for i in range(repeats):
            self._gen_signal(core, preset)
            d_raw, rec, ref = self.run_single_measurement(core, preset)
            if d_raw is None:
                self.log(f"  ✗ Run {i+1}/{repeats}: failed")
                results.append((None, None, None))
            else:
                d_cal = d_raw - calib_total
                self.log(f"  ✓ Run {i+1}/{repeats}: raw {d_raw:.3f} ms -> calibrated {d_cal:.3f} ms")
                results.append((d_cal, rec, ref))
                if first_good is None and rec is not None and ref is not None:
                    first_good = (rec, ref)
            time.sleep(0.2)

        delays = [r[0] for r in results if r[0] is not None]
        avg = safe_mean(delays); std = safe_std(delays)
        if avg is not None:
            self.log(f"  -> average (calibrated): {avg:.3f} ms (std {std:.3f} ms)")

        if save_plot_dir and first_good is not None and avg is not None:
            rec, ref = first_good
            try:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{preset['key']}_plot_{ts}.png"
                out_path = os.path.join(save_plot_dir, fname)
                DelayRunner._save_single_plot(rec, ref, avg, core.sample_rate, out_path, title=f"{preset['name']} (avg {avg:.2f} ms)")
                self.log(f"  Saved plot: {out_path}")
            except Exception as e:
                self.log(f"  (plot save failed: {e})")

        return results

    @staticmethod
    def _save_single_plot(recorded_audio, test_signal, avg_delay_ms, sample_rate, filepath, title="Analysis"):
        plt.figure(figsize=(12, 8))
        plt.suptitle(title, fontsize=14)

        plt.subplot(3, 1, 1)
        t_ref = np.linspace(0, len(test_signal) / sample_rate, len(test_signal), endpoint=False)
        plt.plot(t_ref * 1000.0, test_signal)
        plt.title('Reference'); plt.xlabel('Time (ms)'); plt.ylabel('Amplitude')

        plt.subplot(3, 1, 2)
        t_rec = np.linspace(0, len(recorded_audio) / sample_rate, len(recorded_audio), endpoint=False)
        plt.plot(t_rec * 1000.0, recorded_audio)
        plt.title('Recorded'); plt.xlabel('Time (ms)'); plt.ylabel('Amplitude')

        plt.subplot(3, 1, 3)
        correlation = signal.correlate(recorded_audio, test_signal, mode='full')
        lags = np.arange(-len(test_signal) + 1, len(recorded_audio))
        corr_time = lags / sample_rate * 1000.0
        plt.plot(corr_time, correlation)
        plt.axvline(x=avg_delay_ms, linestyle='--', label=f'Avg Calibrated Delay: {avg_delay_ms:.2f} ms')
        plt.legend(); plt.title('Cross-correlation')
        plt.xlabel('Delay (ms)'); plt.ylabel('Correlation')

        plt.tight_layout(); plt.savefig(filepath); plt.close()

    @staticmethod
    def save_overall_bar(all_delays_by_preset, filepath):
        names = []; means = []; stds = []
        for name, delays in all_delays_by_preset.items():
            d = [x for x in delays if x is not None]
            if not d: continue
            names.append(name); means.append(np.mean(d)); stds.append(np.std(d))
        if not names: return False
        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, means, yerr=stds, capsize=5)
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, y + 1, f"{y:.1f}", ha='center', va='bottom')
        plt.ylabel("Average Calibrated Delay (ms)")
        plt.title("Average Calibrated Delay per Sound")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout(); plt.savefig(filepath); plt.close()
        return True


# -------------------------------
# Lab Tests
# -------------------------------
class LabTests:
    def __init__(self, core: AudioCore, log_fn):
        self.core = core
        self.log = log_fn
        self.results = []   # list of dicts

    # --- Balance (L vs R level) ---
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

    # --- Crosstalk (L->R) ---
    def test_crosstalk(self, freq=1000.0, tone_dur=1.0, settle=0.2, direction="LtoR"):
        if direction == "LtoR":
            self.log("[CROSSTALK] Place mic on LEFT earcup (PRIMARY), click OK")
            messagebox.showinfo("Crosstalk", "Place mic on LEFT earcup (PRIMARY), then click OK")
            sig = self.core.generate_sine(freq=freq, duration=tone_dur)
            rec_primary = self._play_and_record(sig, left_only=True, settle=settle)

            self.log("[CROSSTALK] Now move mic to RIGHT earcup (LEAK), click OK")
            messagebox.showinfo("Crosstalk", "Move mic to RIGHT earcup (LEAK), then click OK")
            self.core.generate_sine(freq=freq, duration=tone_dur)
            rec_leak = self._play_and_record(self.core.test_signal, left_only=True, settle=settle)
        else:
            self.log("[CROSSTALK] Place mic on RIGHT earcup (PRIMARY), click OK")
            messagebox.showinfo("Crosstalk", "Place mic on RIGHT earcup (PRIMARY), then click OK")
            sig = self.core.generate_sine(freq=freq, duration=tone_dur)
            rec_primary = self._play_and_record(sig, right_only=True, settle=settle)

            self.log("[CROSSTALK] Now move mic to LEFT earcup (LEAK), click OK")
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

    # --- Sweep FR (relative) ---
    def test_sweep_fr(self, f0=20, f1=20000, duration=6.0, save_plot_dir=None):
        self.log("[SWEEP FR] Position mic at reference spot, keep still. Running...")
        sig, t = self.core.generate_log_chirp(f0=f0, f1=f1, duration=duration, amp=0.5)
        rec = self._play_and_record(sig, both_channels=True, settle=0.05, rec_dur=duration+0.5)

        # align via xcorr
        delay_ms, _ = AudioCore.find_delay_ms(rec, sig, self.core.sample_rate)
        delay_s = 0.0 if delay_ms is None else delay_ms/1000.0
        shift = int(round(delay_s * self.core.sample_rate))
        rec_aligned = rec[shift: shift + len(sig)] if shift >=0 else rec[:len(sig)]

        # envelope via Hilbert
        if len(rec_aligned) < len(sig):
            pad = len(sig)-len(rec_aligned)
            rec_aligned = np.pad(rec_aligned, (0,pad))
        env = np.abs(signal.hilbert(rec_aligned))
        env = env / (np.max(env)+1e-12)

        # map t -> f
        T = duration
        freqs_inst = f0 * (f1/f0) ** (t / T)

        # bin into log-spaced frequency grid
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

    # --- THD ---
    def test_thd(self, tones=(100, 1000, 6000), tone_dur=1.0, amp=0.6):
        self.log("[THD] Running tones: " + ", ".join(str(x) for x in tones))
        thd_list = []
        for f in tones:
            sig = self.core.generate_sine(freq=f, duration=tone_dur, amp=amp)
            rec = self._play_and_record(sig, both_channels=True, settle=0.05)
            thd = self._compute_thd(rec, f)
            thd_list.append({"freq": f, "thd_percent": thd})
            self.log(f"  f={f} Hz  THD={thd:.3f}%")

        res = {
            "test": "thd",
            "timestamp": self._now(),
            "params": {"tones": list(tones), "tone_dur": tone_dur},
            "metrics": {"items": thd_list}
        }
        self.results.append(res)
        return res

    # --- Isolation (inside vs outside) ---
    def test_isolation(self, noise_dur=2.0, amp=0.4):
        self.log("[ISOLATION] Place mic near earcup seal (inside). Click OK to measure.")
        messagebox.showinfo("Isolation", "Place mic near earcup seal (inside). Click OK to measure.")
        sig = self.core.generate_pink_noise(duration=noise_dur, amp=amp)
        rec_in = self._play_and_record(sig, both_channels=True, settle=0.05)

        self.log("[ISOLATION] Move mic ~5–10 cm outside the cup. Click OK to measure.")
        messagebox.showinfo("Isolation", "Move mic ~5–10 cm outside the cup. Click OK to measure.")
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

    # --- Helpers ---
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
        else:  # both or mono path
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
        # fundamental bin
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

    # --- Export ---
    def export_last(self, out_dir):
        if not self.results:
            raise RuntimeError("No lab test results yet.")
        ensure_dir(out_dir)
        last = self.results[-1]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(out_dir, f"lab_{last['test']}_{ts}")
        # JSON
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(last, f, indent=2, ensure_ascii=False)
        # CSV quick dump if numeric list present
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
        # Also write a CSV summary
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
# GUI
# -------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1080x760")

        self.cfg = load_config()
        self.core = AudioCore(
            sample_rate=self.cfg["last_settings"]["sample_rate"],
            chunk_size=1024,
            duration=self.cfg["last_settings"]["duration"],
            output_device_index=self.cfg["last_settings"]["output_device_index"],
            input_device_index=self.cfg["last_settings"]["input_device_index"],
        )
        self.runner = DelayRunner(self.cfg, self.log)
        self.lab = LabTests(self.core, self.log)

        self._build_ui()
        self._refresh_device_lists()

        self.worker_thread = None
        self.stop_flag = False

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.page_test = ttk.Frame(nb)
        self.page_cal = ttk.Frame(nb)
        self.page_settings = ttk.Frame(nb)
        self.page_lab = ttk.Frame(nb)
        self.page_export = ttk.Frame(nb)

        nb.add(self.page_test, text="Tests")
        nb.add(self.page_cal, text="Calibration")
        nb.add(self.page_lab, text="Lab Tests")
        nb.add(self.page_settings, text="Settings / Devices")
        nb.add(self.page_export, text="Export")

        self._build_test_tab()
        self._build_cal_tab()
        self._build_lab_tab()
        self._build_settings_tab()
        self._build_export_tab()

        frm_log = ttk.Frame(self)
        frm_log.pack(fill="both", side="bottom")
        ttk.Label(frm_log, text="Log").pack(anchor="w")
        self.txt_log = tk.Text(frm_log, height=12)
        self.txt_log.pack(fill="both", expand=True)

    # --- Original Tests Tab ---
    def _build_test_tab(self):
        frm = self.page_test

        top = ttk.Frame(frm)
        top.pack(fill="x", pady=6)

        self.var_repeats = tk.IntVar(value=self.cfg["last_settings"]["repeats"])
        ttk.Label(top, text="Repeats").pack(side="left", padx=(0,6))
        ttk.Spinbox(top, from_=1, to=20, textvariable=self.var_repeats, width=5).pack(side="left")

        ttk.Button(top, text="Run Selected", command=self.on_run_selected).pack(side="left", padx=6)
        ttk.Button(top, text="Run ALL", command=self.on_run_all).pack(side="left", padx=6)

        self.var_save_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Save per-sound example plot", variable=self.var_save_plots).pack(side="left", padx=12)

        self.var_save_overall = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Save overall bar chart", variable=self.var_save_overall).pack(side="left", padx=12)

        self.var_outdir = tk.StringVar(value=os.getcwd())
        ttk.Button(top, text="Choose Output Folder", command=self._choose_outdir).pack(side="right")
        ttk.Entry(top, textvariable=self.var_outdir, width=40).pack(side="right", padx=6)

        mid = ttk.Frame(frm)
        mid.pack(fill="both", expand=True, pady=6)

        self.lst_presets = tk.Listbox(mid, selectmode="extended", height=8)
        for p in SOUND_PRESETS:
            self.lst_presets.insert(tk.END, p["name"])
        self.lst_presets.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        mid.grid_columnconfigure(0, weight=0)
        mid.grid_columnconfigure(1, weight=1)
        mid.grid_rowconfigure(0, weight=1)

        self.txt_summary = tk.Text(mid)
        self.txt_summary.grid(row=0, column=1, sticky="nsew")

        bottom = ttk.Frame(frm)
        bottom.pack(fill="x", pady=6)
        ttk.Button(bottom, text="Save Text Report", command=self.on_save_report).pack(side="left")

    # --- Calibration Tab ---
    def _build_cal_tab(self):
        frm = self.page_cal

        info = ttk.Frame(frm)
        info.pack(fill="x", pady=6)
        ttk.Label(info, text="Per-sound calibration saves a baseline for each preset.\nGlobal system offset (impulse) is recommended with a direct loopback cable."
                 ).pack(anchor="w")

        row1 = ttk.Frame(frm)
        row1.pack(fill="x", pady=6)
        ttk.Button(row1, text="Calibrate Selected Preset", command=self.on_calibrate_selected).pack(side="left", padx=(0,6))
        ttk.Button(row1, text="Calibrate ALL Presets", command=self.on_calibrate_all).pack(side="left", padx=6)

        self.var_cal_repeats = tk.IntVar(value=5)
        ttk.Label(row1, text="Repeats").pack(side="left", padx=(12,6))
        ttk.Spinbox(row1, from_=1, to=20, textvariable=self.var_cal_repeats, width=5).pack(side="left")

        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=6)
        ttk.Button(row2, text="Calibrate GLOBAL System Offset (Impulse)", command=self.on_calibrate_global).pack(side="left", padx=(0,6))
        self.lbl_global = ttk.Label(row2, text=f"Current GLOBAL offset: {self.cfg.get('global_system_offset_ms',0.0):.3f} ms")
        self.lbl_global.pack(side="left", padx=12)

    # --- Lab Tests Tab ---
    def _build_lab_tab(self):
        frm = self.page_lab
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_columnconfigure(1, weight=1)

        # Balance
        grp_bal = ttk.LabelFrame(frm, text="Channel Balance", padding=8)
        grp_bal.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        ttk.Label(grp_bal, text="1 kHz default; prompts will guide mic placement.").grid(row=0, column=0, sticky="w")
        self.var_bal_freq = tk.DoubleVar(value=1000.0)
        ttk.Label(grp_bal, text="Freq (Hz)").grid(row=1, column=0, sticky="w")
        ttk.Entry(grp_bal, textvariable=self.var_bal_freq, width=10).grid(row=1, column=1, sticky="w")
        ttk.Button(grp_bal, text="Run", command=self.on_lab_balance).grid(row=2, column=0, pady=6, sticky="w")

        # Crosstalk
        grp_xt = ttk.LabelFrame(frm, text="Crosstalk", padding=8)
        grp_xt.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        ttk.Label(grp_xt, text="Measures leak in dB (L→R or R→L).").grid(row=0, column=0, columnspan=2, sticky="w")
        self.var_xt_freq = tk.DoubleVar(value=1000.0)
        ttk.Label(grp_xt, text="Freq (Hz)").grid(row=1, column=0, sticky="w")
        ttk.Entry(grp_xt, textvariable=self.var_xt_freq, width=10).grid(row=1, column=1, sticky="w")
        self.var_xt_dir = tk.StringVar(value="LtoR")
        ttk.Radiobutton(grp_xt, text="L→R", variable=self.var_xt_dir, value="LtoR").grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(grp_xt, text="R→L", variable=self.var_xt_dir, value="RtoL").grid(row=2, column=1, sticky="w")
        ttk.Button(grp_xt, text="Run", command=self.on_lab_crosstalk).grid(row=3, column=0, pady=6, sticky="w")

        # Sweep FR
        grp_fr = ttk.LabelFrame(frm, text="Sweep FR (relative)", padding=8)
        grp_fr.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        ttk.Label(grp_fr, text="20–20k logarithmic sweep; saves plot if output folder set.").grid(row=0, column=0, sticky="w")
        self.var_fr_dur = tk.DoubleVar(value=6.0)
        ttk.Label(grp_fr, text="Duration (s)").grid(row=1, column=0, sticky="w")
        ttk.Entry(grp_fr, textvariable=self.var_fr_dur, width=10).grid(row=1, column=1, sticky="w")
        ttk.Button(grp_fr, text="Run", command=self.on_lab_sweep).grid(row=2, column=0, pady=6, sticky="w")

        # THD
        grp_thd = ttk.LabelFrame(frm, text="THD", padding=8)
        grp_thd.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)
        ttk.Label(grp_thd, text="Measures THD at 100 / 1k / 6k Hz.").grid(row=0, column=0, sticky="w")
        ttk.Button(grp_thd, text="Run", command=self.on_lab_thd).grid(row=1, column=0, pady=6, sticky="w")

        # Isolation
        grp_iso = ttk.LabelFrame(frm, text="Isolation (Inside→Outside)", padding=8)
        grp_iso.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        ttk.Label(grp_iso, text="Plays pink noise; measure inside pad then outside ~10 cm.").grid(row=0, column=0, sticky="w")
        ttk.Button(grp_iso, text="Run", command=self.on_lab_isolation).grid(row=1, column=0, pady=6, sticky="w")

        # Lab Results box + export
        grp_res = ttk.LabelFrame(frm, text="Lab Results", padding=8)
        grp_res.grid(row=2, column=1, sticky="nsew", padx=8, pady=8)
        self.txt_lab = tk.Text(grp_res, height=18)
        self.txt_lab.pack(fill="both", expand=True)
        btns = ttk.Frame(grp_res); btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Export LAST (JSON/CSV)", command=self.on_lab_export_last).pack(side="left", padx=4)
        ttk.Button(btns, text="Export ALL (JSON/CSV)", command=self.on_lab_export_all).pack(side="left", padx=4)

    # --- Settings Tab ---
    def _build_settings_tab(self):
        frm = self.page_settings

        af = ttk.LabelFrame(frm, text="Audio Parameters")
        af.pack(fill="x", padx=6, pady=6)
        self.var_sr = tk.IntVar(value=self.cfg["last_settings"]["sample_rate"])
        self.var_dur = tk.DoubleVar(value=self.cfg["last_settings"]["duration"])

        ttk.Label(af, text="Sample Rate").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(af, from_=16000, to=192000, increment=1000, textvariable=self.var_sr, width=8).grid(row=0, column=1, padx=6)

        ttk.Label(af, text="Signal Duration (s)").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(af, from_=0.05, to=2.0, increment=0.05, textvariable=self.var_dur, width=8).grid(row=1, column=1, padx=6)

        ttk.Button(af, text="Apply", command=self.on_apply_audio_params).grid(row=0, column=2, rowspan=2, padx=12)

        df = ttk.LabelFrame(frm, text="Devices")
        df.pack(fill="both", expand=True, padx=6, pady=6)

        ttk.Label(df, text="Output Device").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(df, text="Input Device").grid(row=1, column=0, sticky="w", padx=6, pady=4)

        self.cmb_out = ttk.Combobox(df, state="readonly", width=70)
        self.cmb_in  = ttk.Combobox(df, state="readonly", width=70)
        self.cmb_out.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        self.cmb_in.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        ttk.Button(df, text="Refresh Devices", command=self._refresh_device_lists).grid(row=0, column=2, rowspan=2, padx=6)
        ttk.Button(df, text="Use Selected Devices", command=self.on_use_selected_devices).grid(row=0, column=3, rowspan=2, padx=6)
        df.grid_columnconfigure(1, weight=1)

    # --- Export Tab (legacy tests) ---
    def _build_export_tab(self):
        frm = self.page_export
        ttk.Label(frm, text="Export legacy test results from the Tests tab (summary + plots).").pack(anchor="w", padx=8, pady=(8,2))
        ttk.Button(frm, text="Save Text Report", command=self.on_save_report).pack(anchor="w", padx=8, pady=4)

    # --- Actions (Original Tests) ---
    def on_apply_audio_params(self):
        self.core.sample_rate = int(self.var_sr.get())
        self.core.duration = float(self.var_dur.get())
        self.cfg["last_settings"]["sample_rate"] = self.core.sample_rate
        self.cfg["last_settings"]["duration"] = self.core.duration
        save_config(self.cfg)
        self.log(f"[SET] Applied sample_rate={self.core.sample_rate}, duration={self.core.duration}s")

    def on_use_selected_devices(self):
        out_idx = self.cmb_out.current()
        in_idx = self.cmb_in.current()
        if out_idx < 0 or in_idx < 0:
            messagebox.showwarning("Devices", "Please select both output and input devices.")
            return
        out_dev_idx = self.cmb_out_values[out_idx][0]
        in_dev_idx = self.cmb_in_values[in_idx][0]

        self.core.set_devices(output_index=out_dev_idx, input_index=in_dev_idx)
        self.cfg["last_settings"]["output_device_index"] = out_dev_idx
        self.cfg["last_settings"]["input_device_index"] = in_dev_idx
        save_config(self.cfg)
        self.log(f"[SET] Using output_device_index={out_dev_idx}, input_device_index={in_dev_idx}")

    def on_run_selected(self):
        sel = self.lst_presets.curselection()
        if not sel:
            messagebox.showinfo("Run Selected", "Select at least one preset.")
            return
        presets = [SOUND_PRESETS[i] for i in sel]
        self._start_worker(self._do_run_tests, presets)

    def on_run_all(self):
        self._start_worker(self._do_run_tests, SOUND_PRESETS[:])

    def on_save_report(self):
        if not hasattr(self, "last_all_results"):
            messagebox.showinfo("Save Report", "Run tests first.")
            return
        out_dir = self.var_outdir.get() or os.getcwd()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rep_path = os.path.join(out_dir, f"latency_report_{ts}.txt")
        # Minimal legacy report (summary already shows stats)
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write("Latency Test Report\n")
            f.write(f"Date: {datetime.datetime.now()}\n\n")
            for name, data in self.last_all_results.items():
                dd = data["delays_ms"]
                if dd:
                    f.write(f"{name}: avg {np.mean(dd):.3f} ms, std {np.std(dd):.3f}, n {len(dd)}\n")
                else:
                    f.write(f"{name}: no successful runs\n")
        messagebox.showinfo("Report", f"Saved report:\n{rep_path}")
        self.log(f"[SAVE] Report -> {rep_path}")

    def on_calibrate_selected(self):
        sel = self.lst_presets.curselection()
        if not sel:
            messagebox.showinfo("Calibration", "Select at least one preset (left list on Tests tab).")
            return
        presets = [SOUND_PRESETS[i] for i in sel]
        self._start_worker(self._do_calibrate_presets, presets)

    def on_calibrate_all(self):
        self._start_worker(self._do_calibrate_presets, SOUND_PRESETS[:])

    def on_calibrate_global(self):
        self._start_worker(self._do_calibrate_global)

    def _choose_outdir(self):
        d = filedialog.askdirectory(title="Choose output folder", mustexist=True)
        if d:
            self.var_outdir.set(d)

    def _start_worker(self, fn, *args):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "Another task is running.")
            return
        self.worker_thread = threading.Thread(target=self._guarded_run, args=(fn, *args), daemon=True)
        self.worker_thread.start()

    def _guarded_run(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def _do_run_tests(self, presets):
        repeats = int(self.var_repeats.get())
        out_dir = self.var_outdir.get() or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        self.txt_summary.delete("1.0", tk.END)
        all_results = {}
        all_delays_bar = {}

        for p in presets:
            results = self.runner.run_test_for_preset(
                self.core, p, repeats=repeats,
                save_plot_dir=out_dir if self.var_save_plots.get() else None
            )
            delays = [r[0] for r in results if r[0] is not None]
            all_results[p["name"]] = {
                "per_sound_baseline_ms": self.cfg["per_sound_offsets_ms"].get(p["key"], 0.0),
                "delays_ms": delays
            }
            all_delays_bar[p["name"]] = delays

        self.last_all_results = all_results

        lines = ["=== SUMMARY (Calibrated) ==="]
        for name, data in all_results.items():
            d = data["delays_ms"]
            if d:
                lines.append(f"{name}: avg {np.mean(d):.2f} ms | std {np.std(d):.2f} | n {len(d)}")
            else:
                lines.append(f"{name}: no successful runs")
        self.txt_summary.insert("1.0", "\n".join(lines))

        if self.var_save_overall.get():
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bar_path = os.path.join(out_dir, f"overall_bar_{ts}.png")
            ok = DelayRunner.save_overall_bar(all_delays_by_preset=all_delays_bar, filepath=bar_path)
            if ok:
                self.log(f"[SAVE] Overall bar chart -> {bar_path}")
            else:
                self.log("[SAVE] Overall bar chart skipped (no data).")

        self.log("[DONE] Test sequence complete.")

    def _do_calibrate_presets(self, presets):
        repeats = int(self.var_cal_repeats.get())
        for p in presets:
            self.runner.run_calibration_for_preset(self.core, p, repeats=repeats)
        self.log("[DONE] Per-sound calibration complete.")

    def _do_calibrate_global(self):
        repeats = int(self.var_cal_repeats.get())
        val = self.runner.run_global_system_calibration(self.core, repeats=repeats)
        if val is not None:
            self.lbl_global.config(text=f"Current GLOBAL offset: {val:.3f} ms")
        self.log("[DONE] Global calibration complete.")

    # --- Lab Actions ---
    def on_lab_balance(self):
        try:
            freq = float(self.var_bal_freq.get())
            res = self.lab.test_balance(freq=freq)
            self._append_lab_result(res)
        except Exception as e:
            messagebox.showerror("Balance", str(e))

    def on_lab_crosstalk(self):
        try:
            freq = float(self.var_xt_freq.get())
            direction = self.var_xt_dir.get()
            res = self.lab.test_crosstalk(freq=freq, direction=direction)
            self._append_lab_result(res)
        except Exception as e:
            messagebox.showerror("Crosstalk", str(e))

    def on_lab_sweep(self):
        try:
            dur = float(self.var_fr_dur.get())
            out_dir = self.var_outdir.get() or os.getcwd()
            res = self.lab.test_sweep_fr(duration=dur, save_plot_dir=out_dir)
            self._append_lab_result(res)
        except Exception as e:
            messagebox.showerror("Sweep FR", str(e))

    def on_lab_thd(self):
        try:
            res = self.lab.test_thd()
            self._append_lab_result(res)
        except Exception as e:
            messagebox.showerror("THD", str(e))

    def on_lab_isolation(self):
        try:
            res = self.lab.test_isolation()
            self._append_lab_result(res)
        except Exception as e:
            messagebox.showerror("Isolation", str(e))

    def on_lab_export_last(self):
        try:
            out_dir = self.var_outdir.get() or os.getcwd()
            path = self.lab.export_last(out_dir)
            self.log(f"[EXPORT] Last lab result -> {path}")
            messagebox.showinfo("Export", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export", str(e))

    def on_lab_export_all(self):
        try:
            out_dir = self.var_outdir.get() or os.getcwd()
            path = self.lab.export_all(out_dir)
            self.log(f"[EXPORT] All lab results -> {path}")
            messagebox.showinfo("Export", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export", str(e))

    def _append_lab_result(self, res: dict):
        self.txt_lab.insert("end", json.dumps(res, indent=2, ensure_ascii=False) + "\n\n")
        self.txt_lab.see("end")

    # --- Devices ---
    def _refresh_device_lists(self):
        devs = self.core.list_devices()
        out_entries = []
        in_entries = []
        for i, info in enumerate(devs):
            name = info.get("name","unknown")
            max_in = int(info.get("maxInputChannels", 0))
            max_out = int(info.get("maxOutputChannels", 0))
            rate = int(info.get("defaultSampleRate", 0))
            label = f"[{i}] {name}  (in:{max_in}, out:{max_out}, rate:{rate})"
            if max_out > 0:
                out_entries.append((i, label))
            if max_in > 0:
                in_entries.append((i, label))

        self.cmb_out_values = out_entries
        self.cmb_in_values = in_entries

        self.cmb_out['values'] = [x[1] for x in out_entries]
        self.cmb_in['values'] = [x[1] for x in in_entries]

        last_out = self.cfg["last_settings"].get("output_device_index")
        last_in = self.cfg["last_settings"].get("input_device_index")

        if last_out is not None:
            for idx, (dev_index, _) in enumerate(out_entries):
                if dev_index == last_out:
                    self.cmb_out.current(idx); break
        if last_in is not None:
            for idx, (dev_index, _) in enumerate(in_entries):
                if dev_index == last_in:
                    self.cmb_in.current(idx); break

    def log(self, msg):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)


if __name__ == "__main__":
    app = App()
    app.mainloop()
