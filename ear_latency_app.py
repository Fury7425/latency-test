
import os
import json
import time
import threading
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Core scientific/audio deps
import numpy as np
from scipy import signal

# PyAudio is used exactly like in your scripts
import pyaudio

# Optional plotting (saved to files only)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


APP_TITLE = "Ear/Headset Latency Tester"
CALIBRATION_JSON = "calibration_per_sound.json"  # per-sound + global offsets


# -------------------------------
# Utility
# -------------------------------
def safe_mean(a):
    vals = [x for x in a if x is not None]
    return float(np.mean(vals)) if vals else None

def safe_std(a):
    vals = [x for x in a if x is not None]
    return float(np.std(vals)) if vals else None


# -------------------------------
# Audio Core (play, record, analyze)
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

    def generate_sine(self, freq=1000.0, duration=None):
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
        sig = np.sin(2 * np.pi * freq * t) * envelope
        self.test_signal = sig.astype(np.float32)
        return self.test_signal

    def generate_impulse(self, buffer_duration=0.1, impulse_duration_samples=1, amplitude=0.8):
        num_samples = int(self.sample_rate * buffer_duration)
        sig = np.zeros(num_samples, dtype=np.float32)
        start = max(0, num_samples // 2 - impulse_duration_samples // 2)
        end = min(num_samples, start + impulse_duration_samples)
        sig[start:end] = amplitude
        self.test_signal = sig
        return self.test_signal

    def play_signal(self):
        if self.test_signal is None:
            raise RuntimeError("No test_signal. Generate first.")
        audio_data = (self.test_signal * 32767).astype(np.int16)
        self.stream_output = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.output_device_index
        )
        self.stream_output.write(audio_data.tobytes())
        self.stream_output.stop_stream()
        self.stream_output.close()
        self.stream_output = None

    def record_audio(self, duration):
        self.stream_input = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.input_device_index
        )
        frames = []
        num_chunks = int(self.sample_rate * duration / self.chunk_size)
        for _ in range(num_chunks):
            try:
                data = self.stream_input.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
            except IOError:
                break

        self.stream_input.stop_stream()
        self.stream_input.close()
        self.stream_input = None

        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32767.0
        self.recorded_audio = audio_data
        return self.recorded_audio

    @staticmethod
    def find_delay_ms(recorded_signal, reference_signal, sample_rate):
        """ Cross-correlation peak + parabolic interpolation (same idea as user's code). """
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
            if self.stream_input:
                if self.stream_input.is_active():
                    self.stream_input.stop_stream()
                self.stream_input.close()
                self.stream_input = None
            if self.stream_output:
                if self.stream_output.is_active():
                    self.stream_output.stop_stream()
                self.stream_output.close()
                self.stream_output = None
            if self.audio:
                self.audio.terminate()
                self.audio = None
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
        "global_system_offset_ms": 0.0,   # e.g., measured via direct loopback
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
            # Fill defaults if keys missing
            cfg = default_config()
            cfg.update(data)
            # ensure per_sound_offsets contains all keys
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
# Test/Calibration Runner
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
        """Play, record, find raw delay in ms."""
        # record thread
        rec_dur = float(core.duration) + float(record_margin_s)
        rec_data = {"audio": None}

        def record_worker():
            rec_data["audio"] = core.record_audio(rec_dur)

        t = threading.Thread(target=record_worker, daemon=True)
        t.start()
        time.sleep(0.1)  # ensure recording started

        core.play_signal()
        t.join()

        delay_ms, _ = AudioCore.find_delay_ms(rec_data["audio"], core.test_signal, core.sample_rate)
        return delay_ms, rec_data["audio"], core.test_signal

    def run_calibration_for_preset(self, core: AudioCore, preset, repeats=5):
        self.log(f"[CAL] {preset['name']} - measuring baseline offset ({repeats} runs) ...")
        delays = []
        for i in range(repeats):
            self._gen_signal(core, preset)
            d, _, _ = self.run_single_measurement(core, preset)
            if d is None:
                self.log(f"  ✗ Run {i+1}/{repeats}: failed (no audio)")
            else:
                self.log(f"  ✓ Run {i+1}/{repeats}: raw delay {d:.3f} ms")
            delays.append(d)
            time.sleep(0.3)

        avg = safe_mean(delays)
        std = safe_std(delays)
        if avg is None:
            self.log("  -> No successful calibration runs.")
            return None
        self.cfg["per_sound_offsets_ms"][preset["key"]] = float(avg)
        save_config(self.cfg)
        self.log(f"  -> Saved per-sound baseline for {preset['name']}: {avg:.3f} ms (std {std:.3f} ms)")
        return avg

    def run_global_system_calibration(self, core: AudioCore, repeats=10):
        """Use impulse by default; recommend direct loopback for accuracy."""
        pseudo_preset = {"key": "impulse", "name": "System (Impulse)", "type": "impulse", "freq": None}
        self.log(f"[CAL] Global system offset via impulse - {repeats} runs. (Recommend direct loopback cable)")
        delays = []
        for i in range(repeats):
            self._gen_signal(core, pseudo_preset)
            d, _, _ = self.run_single_measurement(core, pseudo_preset)
            if d is None:
                self.log(f"  ✗ Run {i+1}/{repeats}: failed (no audio)")
            else:
                self.log(f"  ✓ Run {i+1}/{repeats}: raw delay {d:.3f} ms")
            delays.append(d)
            time.sleep(0.3)

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
            time.sleep(0.3)

        delays = [r[0] for r in results if r[0] is not None]
        avg = safe_mean(delays)
        std = safe_std(delays)

        if avg is not None:
            self.log(f"  -> {preset['name']} average (calibrated): {avg:.3f} ms (std {std:.3f} ms)")
        else:
            self.log(f"  -> {preset['name']}: no successful runs.")

        # optional plot of first good run + avg line
        if save_plot_dir and first_good is not None and avg is not None:
            rec, ref = first_good
            try:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{preset['key']}_plot_{ts}.png"
                out_path = os.path.join(save_plot_dir, fname)
                self._save_single_plot(rec, ref, avg, core.sample_rate, out_path, title=f"{preset['name']} (avg {avg:.2f} ms)")
                self.log(f"  Saved plot: {out_path}")
            except Exception as e:
                self.log(f"  (plot save failed: {e})")

        return results

    @staticmethod
    def _save_single_plot(recorded_audio, test_signal, avg_delay_ms, sample_rate, filepath, title="Analysis"):
        plt.figure(figsize=(12, 8))
        plt.suptitle(title, fontsize=14)

        # Reference
        plt.subplot(3, 1, 1)
        t_ref = np.linspace(0, len(test_signal) / sample_rate, len(test_signal), endpoint=False)
        plt.plot(t_ref * 1000.0, test_signal)
        plt.title('Reference')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')

        # Recorded
        plt.subplot(3, 1, 2)
        t_rec = np.linspace(0, len(recorded_audio) / sample_rate, len(recorded_audio), endpoint=False)
        plt.plot(t_rec * 1000.0, recorded_audio)
        plt.title('Recorded')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')

        # Xcorr
        plt.subplot(3, 1, 3)
        correlation = signal.correlate(recorded_audio, test_signal, mode='full')
        lags = np.arange(-len(test_signal) + 1, len(recorded_audio))
        corr_time = lags / sample_rate * 1000.0
        plt.plot(corr_time, correlation)
        plt.axvline(x=avg_delay_ms, linestyle='--', label=f'Avg Calibrated Delay: {avg_delay_ms:.2f} ms')
        plt.legend()
        plt.title('Cross-correlation')
        plt.xlabel('Delay (ms)')
        plt.ylabel('Correlation')

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    @staticmethod
    def save_overall_bar(all_delays_by_preset, filepath):
        names = []
        means = []
        stds = []
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
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, y + 1, f"{y:.1f}", ha='center', va='bottom')
        plt.ylabel("Average Calibrated Delay (ms)")
        plt.title("Average Calibrated Delay per Sound")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        return True

    @staticmethod
    def save_report(all_results, cfg, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("="*64 + "\n")
            f.write("HEADSET LATENCY TEST REPORT\n")
            f.write("="*64 + "\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Calibration used: GLOBAL {cfg.get('global_system_offset_ms',0.0):.3f} ms + per-sound (see below)\n\n")

            overall = []
            for name, data in all_results.items():
                f.write(f"\n--- {name} ---\n")
                f.write(f"Per-sound baseline: {data['per_sound_baseline_ms']:.3f} ms\n")
                delays = data["delays_ms"]
                good = [x for x in delays if x is not None]
                if good:
                    f.write("Individual calibrated delays (ms): " + ", ".join(f"{x:.3f}" for x in good) + "\n")
                    f.write(f"Average: {np.mean(good):.3f} ms | Std: {np.std(good):.3f} ms | Min: {np.min(good):.3f} ms | Max: {np.max(good):.3f} ms\n")
                    overall.extend(good)
                else:
                    f.write("No successful runs.\n")

            if overall:
                f.write("\n=== OVERALL ===\n")
                f.write(f"Overall average: {np.mean(overall):.3f} ms | Std: {np.std(overall):.3f} ms | N: {len(overall)}\n")
            f.write("\nREPORT END\n")


# -------------------------------
# GUI
# -------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x720")

        self.cfg = load_config()
        self.core = AudioCore(
            sample_rate=self.cfg["last_settings"]["sample_rate"],
            chunk_size=1024,
            duration=self.cfg["last_settings"]["duration"],
            output_device_index=self.cfg["last_settings"]["output_device_index"],
            input_device_index=self.cfg["last_settings"]["input_device_index"],
        )
        self.runner = DelayRunner(self.cfg, self.log)

        self._build_ui()
        self._refresh_device_lists()

        self.worker_thread = None
        self.stop_flag = False

    # ---------- UI ---------
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.page_test = ttk.Frame(nb)
        self.page_cal = ttk.Frame(nb)
        self.page_settings = ttk.Frame(nb)

        nb.add(self.page_test, text="Tests")
        nb.add(self.page_cal, text="Calibration")
        nb.add(self.page_settings, text="Settings / Devices")

        self._build_test_tab()
        self._build_cal_tab()
        self._build_settings_tab()

        # Log
        frm_log = ttk.Frame(self)
        frm_log.pack(fill="both", side="bottom")
        ttk.Label(frm_log, text="Log").pack(anchor="w")
        self.txt_log = tk.Text(frm_log, height=12)
        self.txt_log.pack(fill="both", expand=True)

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

        row3 = ttk.Frame(frm)
        row3.pack(fill="both", expand=True, pady=6)
        self.tree_per_sound = ttk.Treeview(row3, columns=("baseline",), show="headings", height=8)
        self.tree_per_sound.heading("baseline", text="Per-sound baseline (ms)")
        self.tree_per_sound.grid(row=0, column=0, sticky="nsew")
        row3.grid_columnconfigure(0, weight=1)
        row3.grid_rowconfigure(0, weight=1)

        self._refresh_per_sound_table()

    def _build_settings_tab(self):
        frm = self.page_settings

        # Audio params
        af = ttk.LabelFrame(frm, text="Audio Parameters")
        af.pack(fill="x", padx=6, pady=6)
        self.var_sr = tk.IntVar(value=self.cfg["last_settings"]["sample_rate"])
        self.var_dur = tk.DoubleVar(value=self.cfg["last_settings"]["duration"])

        ttk.Label(af, text="Sample Rate").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(af, from_=16000, to=192000, increment=1000, textvariable=self.var_sr, width=8).grid(row=0, column=1, padx=6)

        ttk.Label(af, text="Signal Duration (s)").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(af, from_=0.05, to=2.0, increment=0.05, textvariable=self.var_dur, width=8).grid(row=1, column=1, padx=6)

        ttk.Button(af, text="Apply", command=self.on_apply_audio_params).grid(row=0, column=2, rowspan=2, padx=12)

        # Devices
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

    # ---------- Actions ---------
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
        DelayRunner.save_report(self.last_all_results, self.cfg, rep_path)
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

    # ---------- Worker dispatch ---------
    def _start_worker(self, fn, *args):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "Another task is running.")
            return
        self.stop_flag = False
        self.worker_thread = threading.Thread(target=self._guarded_run, args=(fn, *args), daemon=True)
        self.worker_thread.start()

    def _guarded_run(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            self.log(f"[ERROR] {e}")
        finally:
            pass

    # ---------- Worker implementations ---------
    def _do_run_tests(self, presets):
        repeats = int(self.var_repeats.get())
        out_dir = self.var_outdir.get() or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        self.txt_summary.delete("1.0", tk.END)
        all_results = {}
        all_delays_bar = {}

        for p in presets:
            # Run preset tests
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

        # Write brief summary
        lines = []
        lines.append("=== SUMMARY (Calibrated) ===")
        for name, data in all_results.items():
            d = data["delays_ms"]
            if d:
                lines.append(f"{name}: avg {np.mean(d):.2f} ms | std {np.std(d):.2f} | n {len(d)}")
            else:
                lines.append(f"{name}: no successful runs")
        self.txt_summary.insert("1.0", "\n".join(lines))

        # Overall bar chart
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
        self._refresh_per_sound_table()
        self.log("[DONE] Per-sound calibration complete.")

    def _do_calibrate_global(self):
        repeats = int(self.var_cal_repeats.get())
        val = self.runner.run_global_system_calibration(self.core, repeats=repeats)
        if val is not None:
            self.lbl_global.config(text=f"Current GLOBAL offset: {val:.3f} ms")
        self.log("[DONE] Global calibration complete.")

    # ---------- Helpers ---------
    def _refresh_per_sound_table(self):
        for i in self.tree_per_sound.get_children():
            self.tree_per_sound.delete(i)
        for p in SOUND_PRESETS:
            val = self.cfg["per_sound_offsets_ms"].get(p["key"], 0.0)
            self.tree_per_sound.insert("", "end", values=(f"{p['name']}",))
            # Treeview headings only show defined columns; we'll prepend name in text
        # Improve: show name + baseline; workaround: use two columns (name, baseline)
        self.tree_per_sound.configure(columns=("name","baseline"))
        self.tree_per_sound.heading("name", text="Preset")
        self.tree_per_sound.heading("baseline", text="Per-sound baseline (ms)")
        for i in self.tree_per_sound.get_children():
            self.tree_per_sound.delete(i)
        for p in SOUND_PRESETS:
            val = self.cfg["per_sound_offsets_ms"].get(p["key"], 0.0)
            self.tree_per_sound.insert("", "end", values=(p["name"], f"{val:.3f}"))

    def _refresh_device_lists(self):
        # Build lists for comboboxes
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

        # Try to preselect last-used devices
        last_out = self.cfg["last_settings"].get("output_device_index")
        last_in = self.cfg["last_settings"].get("input_device_index")

        if last_out is not None:
            for idx, (dev_index, _) in enumerate(out_entries):
                if dev_index == last_out:
                    self.cmb_out.current(idx)
                    break

        if last_in is not None:
            for idx, (dev_index, _) in enumerate(in_entries):
                if dev_index == last_in:
                    self.cmb_in.current(idx)
                    break

    def log(self, msg):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)


if __name__ == "__main__":
    app = App()
    app.mainloop()
