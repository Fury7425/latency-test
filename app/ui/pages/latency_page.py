# app/ui/pages/latency_page.py
import os
import datetime
import numpy as np
import customtkinter as ctk

# ---- Robust imports for both package and flat layouts ----
try:
    from ...tests.delay import DelayRunner, SOUND_PRESETS
    from ...core.utils import ensure_dir
    from ...config import save_config
    from ..glass import GlassCard, resource_path
except ImportError:
    from tests.delay import DelayRunner, SOUND_PRESETS
    from core.utils import ensure_dir
    from config import save_config
    from app.ui.glass import GlassCard, resource_path
# ----------------------------------------------------------


class LatencyPage(ctk.CTkScrollableFrame):
    """
    Refactored Latency page:
    - Glassmorphism layout (cards)
    - Latency test controls + KPIs
    - Calibration (per-preset + global) on same page
    - Summary & Baseline panels
    """
    def __init__(self, master, core, cfg, log_fn):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self.core = core
        self.cfg = cfg
        self.log = log_fn
        self.runner = DelayRunner(self.cfg, self.log)
        self.worker = None

        # Grid for page
        self.grid_columnconfigure(0, weight=1)

        # ---------- HEADER ----------
        hdr = GlassCard(self); hdr.grid(row=0, column=0, padx=18, pady=(16, 8), sticky="ew")
        row = hdr.inner
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text="Latency", font=ctk.CTkFont(size=20, weight="bold")).grid(
            row=0, column=0, padx=14, pady=(12, 6), sticky="w"
        )
        ctk.CTkLabel(row, text="Measure end‑to‑end latency and manage calibration baselines.",
                     font=ctk.CTkFont(size=12)).grid(row=1, column=0, columnspan=2, padx=14, pady=(0, 12), sticky="w")

        # ---------- RUN CARD ----------
        card_run = GlassCard(self); card_run.grid(row=1, column=0, padx=18, pady=8, sticky="ew")
        run = card_run.inner
        for c in range(8): run.grid_columnconfigure(c, weight=1)

        ctk.CTkLabel(run, text="Run Delay Tests", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=8, padx=14, pady=(12, 2), sticky="w"
        )

        # Preset toggles (compact grid)
        self.preset_vars = {}
        r_base = 1
        for i, p in enumerate(SOUND_PRESETS):
            var = ctk.BooleanVar(value=(p["key"] != "impulse"))
            self.preset_vars[p["key"]] = var
            ctk.CTkSwitch(run, text=p["name"], variable=var).grid(
                row=r_base + i // 4, column=i % 4, padx=10, pady=6, sticky="w"
            )

        # Repeats slider
        ctk.CTkLabel(run, text="Repeats").grid(row=r_base + 2, column=0, padx=14, pady=(8, 2), sticky="w")
        self.repeats_slider = ctk.CTkSlider(run, from_=1, to=20, number_of_steps=19, width=240)
        self.repeats_slider.set(int(self.cfg["last_settings"].get("repeats", 5)))
        self.repeats_slider.grid(row=r_base + 2, column=1, padx=(6, 14), pady=(8, 2), sticky="w")
        self._rep_val = ctk.StringVar(value=str(int(self.repeats_slider.get())))
        self.repeats_label = ctk.CTkLabel(run, textvariable=self._rep_val)
        self.repeats_label.grid(row=r_base + 2, column=2, padx=4, pady=(8, 2), sticky="w")
        self.repeats_slider.configure(command=lambda v: self._rep_val.set(str(int(float(v)))))

        # Output dir
        ctk.CTkLabel(run, text="Output Folder").grid(row=r_base + 3, column=0, padx=14, pady=(8, 2), sticky="w")
        self.output_dir_var = ctk.StringVar(value=self.cfg["last_settings"].get("output_dir", os.getcwd()))
        out_row = ctk.CTkFrame(run, fg_color="transparent"); out_row.grid(
            row=r_base + 3, column=1, columnspan=3, padx=(4, 14), pady=(6, 12), sticky="ew"
        )
        out_row.grid_columnconfigure(0, weight=1)
        self.output_entry = ctk.CTkEntry(out_row, textvariable=self.output_dir_var)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(out_row, text="Browse", width=90, command=self._choose_outdir).grid(row=0, column=1)

        # Save options
        self.save_plots_var = ctk.BooleanVar(value=True)
        self.save_bar_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(run, text="Save per‑sound plot", variable=self.save_plots_var).grid(
            row=r_base + 4, column=0, padx=14, pady=2, sticky="w"
        )
        ctk.CTkSwitch(run, text="Save overall bar chart", variable=self.save_bar_var).grid(
            row=r_base + 4, column=1, padx=6, pady=2, sticky="w"
        )

        # Buttons
        btns = ctk.CTkFrame(run, fg_color="transparent"); btns.grid(
            row=r_base + 5, column=0, columnspan=8, padx=12, pady=(8, 12), sticky="ew"
        )
        ctk.CTkButton(btns, text="Run Selected", command=self.on_run_selected).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Run ALL", command=self.on_run_all).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Save Text Report", command=self.on_save_report).pack(side="right", padx=6)

        # ---------- KPI CARD ----------
        card_kpi = GlassCard(self); card_kpi.grid(row=2, column=0, padx=18, pady=8, sticky="ew")
        kpi = card_kpi.inner
        for i in range(3): kpi.grid_columnconfigure(i, weight=1)
        self.kpi_avg = self._kpi(kpi, "Average (ms)", "--", 0)
        self.kpi_std = self._kpi(kpi, "Std Dev (ms)", "--", 1)
        self.kpi_last = self._kpi(kpi, "Last (ms)", "--", 2)

        # ---------- SUMMARY CARD ----------
        card_sum = GlassCard(self); card_sum.grid(row=3, column=0, padx=18, pady=8, sticky="ew")
        sumf = card_sum.inner
        ctk.CTkLabel(sumf, text="Summary", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=14, pady=(12, 6), sticky="w"
        )
        self.summary_box = ctk.CTkTextbox(sumf, wrap="word", height=160)
        self.summary_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")

        # ---------- CALIBRATION CARD ----------
        card_cal = GlassCard(self); card_cal.grid(row=4, column=0, padx=18, pady=8, sticky="ew")
        cal = card_cal.inner
        for c in range(6): cal.grid_columnconfigure(c, weight=1)
        ctk.CTkLabel(cal, text="Calibration", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=6, padx=14, pady=(12, 2), sticky="w"
        )

        # Per-preset switches
        self.cal_vars = {}
        for i, p in enumerate(SOUND_PRESETS):
            var = ctk.BooleanVar(value=True)
            self.cal_vars[p["key"]] = var
            ctk.CTkSwitch(cal, text=p["name"], variable=var).grid(
                row=1 + i // 4, column=i % 4, padx=10, pady=6, sticky="w"
            )

        # Calibration repeats
        ctk.CTkLabel(cal, text="Repeats").grid(row=3, column=0, padx=14, pady=(8, 2), sticky="w")
        self.cal_repeats_slider = ctk.CTkSlider(cal, from_=1, to=20, number_of_steps=19, width=200)
        self.cal_repeats_slider.set(5)
        self.cal_repeats_slider.grid(row=3, column=1, padx=(6, 14), pady=(8, 2), sticky="w")
        self._cal_rep_val = ctk.StringVar(value="5")
        ctk.CTkLabel(cal, textvariable=self._cal_rep_val).grid(row=3, column=2, padx=4, pady=(8, 2), sticky="w")
        self.cal_repeats_slider.configure(command=lambda v: self._cal_rep_val.set(str(int(float(v)))))

        # Calibration buttons
        cal_btns = ctk.CTkFrame(cal, fg_color="transparent"); cal_btns.grid(
            row=4, column=0, columnspan=6, padx=12, pady=(6, 12), sticky="ew"
        )
        ctk.CTkButton(cal_btns, text="Calibrate Selected Presets", command=self.on_cal_selected).pack(side="left", padx=6)
        ctk.CTkButton(cal_btns, text="Calibrate ALL Presets", command=self.on_cal_all).pack(side="left", padx=6)
        ctk.CTkButton(cal_btns, text="Calibrate GLOBAL (Impulse)", command=self.on_cal_global).pack(side="left", padx=6)

        # Baselines
        card_base = GlassCard(self); card_base.grid(row=5, column=0, padx=18, pady=(8, 18), sticky="ew")
        base = card_base.inner
        ctk.CTkLabel(base, text="Baselines", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=14, pady=(12, 6), sticky="w"
        )
        self.baseline_box = ctk.CTkTextbox(base, wrap="word", height=140)
        self.baseline_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")
        self._refresh_baseline_box()

    # ------------------ UI HELPERS ------------------
    def _kpi(self, parent, title, value, col):
        card = ctk.CTkFrame(parent, fg_color="transparent")
        card.grid(row=0, column=col, padx=8, pady=(8, 12), sticky="ew")
        ctk.CTkLabel(card, text=title).pack(anchor="w", padx=8, pady=(4, 0))
        val = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=22, weight="bold"))
        val.pack(anchor="w", padx=8, pady=(2, 6))
        return val

    def _choose_outdir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory()
        if d:
            self.output_dir_var.set(d)
            self.cfg["last_settings"]["output_dir"] = d
            save_config(self.cfg)

    # ------------------ EVENT HANDLERS ------------------
    def on_run_selected(self):
        presets = [p for p in SOUND_PRESETS if self.preset_vars[p["key"]].get()]
        if presets:
            self._start(self._run_tests, presets)

    def on_run_all(self):
        self._start(self._run_tests, SOUND_PRESETS[:])

    def on_save_report(self):
        if not hasattr(self, "last_all_results"): return
        out_dir = self.output_dir_var.get() or os.getcwd()
        ensure_dir(out_dir)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"latency_report_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Latency Report\n")
            for name, data in self.last_all_results.items():
                dd = data["delays_ms"]
                if dd:
                    f.write(f"{name}: avg {np.mean(dd):.3f} ms, std {np.std(dd):.3f}, n {len(dd)}\n")
                else:
                    f.write(f"{name}: no successful runs\n")
        self.log(f"[SAVE] report -> {path}")

    def on_cal_selected(self):
        presets = [p for p in SOUND_PRESETS if self.cal_vars[p["key"]].get()]
        if presets:
            self._start(self._cal_presets, presets)

    def on_cal_all(self):
        self._start(self._cal_presets, SOUND_PRESETS[:])

    def on_cal_global(self):
        self._start(self._cal_global)

    # ------------------ WORKER WRAPPERS ------------------
    def _start(self, fn, *args):
        import threading
        if getattr(self, "worker", None) and self.worker.is_alive():
            self.log("[BUSY] another task is running."); return
        self.worker = threading.Thread(target=self._guard, args=(fn, *args), daemon=True)
        self.worker.start()

    def _guard(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            self.log(f"[ERROR] {e}")

    # ------------------ CORE OPS ------------------
    def _run_tests(self, presets):
        repeats = int(float(self._rep_val.get()))
        out_dir = self.output_dir_var.get() or os.getcwd()
        ensure_dir(out_dir)

        all_results = {}
        overall = []
        last = None
        bars = {}

        for p in presets:
            results = self.runner.run_test(
                self.core, p, repeats=repeats,
                save_plot_dir=out_dir if self.save_plots_var.get() else None
            )
            delays = [r for r in results if r is not None]
            all_results[p["name"]] = {
                "per_sound_baseline_ms": self.cfg["per_sound_offsets_ms"].get(p["key"], 0.0),
                "delays_ms": delays
            }
            bars[p["name"]] = delays
            if delays:
                overall.extend(delays)
                last = delays[-1]

        self.last_all_results = all_results
        self._refresh_summary(all_results)

        if overall:
            self.kpi_avg.configure(text=f"{np.mean(overall):.1f}")
            self.kpi_std.configure(text=f"{np.std(overall):.1f}")
            self.kpi_last.configure(text=f"{last:.1f}")
        else:
            self.kpi_avg.configure(text="--")
            self.kpi_std.configure(text="--")
            self.kpi_last.configure(text="--")

        # Overall bar chart
        if self.save_bar_var.get():
            try:
                # reuse static helper if present
                from ...tests.delay import DelayRunner as _DR  # package path
            except Exception:
                from tests.delay import DelayRunner as _DR       # flat path
            try:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                bar_path = os.path.join(out_dir, f"overall_bar_{ts}.png")
                ok = _safe_save_bar(_DR, bars, bar_path)
                if ok:
                    self.log(f"[SAVE] bar chart -> {bar_path}")
            except Exception as e:
                self.log(f"[WARN] bar chart failed: {e}")

        self.log("[DONE] tests complete.")

    def _cal_presets(self, presets):
        reps = int(float(self._cal_rep_val.get()))
        for p in presets:
            self.runner.calibrate_preset(self.core, p, repeats=reps)
        save_config(self.cfg)
        self._refresh_baseline_box()
        self.log("[DONE] per‑sound calibration complete.")

    def _cal_global(self):
        reps = int(float(self._cal_rep_val.get()))
        val, _ = self.runner.calibrate_global(self.core, repeats=reps)
        if val is not None:
            self.log(f"[CAL] GLOBAL set to {val:.2f} ms")
        save_config(self.cfg)
        self._refresh_baseline_box()
        self.log("[DONE] global calibration complete.")

    # ------------------ TEXT PANELS ------------------
    def _refresh_summary(self, all_results):
        lines = ["=== SUMMARY (Calibrated) ==="]
        for name, data in all_results.items():
            dd = data["delays_ms"]
            if dd:
                lines.append(f"{name}: avg {np.mean(dd):.2f} ms | std {np.std(dd):.2f} | n {len(dd)}")
            else:
                lines.append(f"{name}: no successful runs")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", "\n".join(lines))

    def _refresh_baseline_box(self):
        lines = ["Per‑sound baselines (ms):"]
        for p in SOUND_PRESETS:
            val = self.cfg["per_sound_offsets_ms"].get(p["key"], 0.0)
            lines.append(f"• {p['name']}: {val:.2f}")
        lines.append(f"\nGLOBAL offset: {self.cfg.get('global_system_offset_ms', 0.0):.2f} ms")
        self.baseline_box.delete("1.0", "end")
        self.baseline_box.insert("1.0", "\n".join(lines))


# ------------- helpers -------------
def _safe_save_bar(DelayRunnerClass, bars_dict, out_path):
    """
    Try to call DelayRunner.save_bar(bars, path) if available.
    Returns True if a file was saved.
    """
    if hasattr(DelayRunnerClass, "save_bar"):
        try:
            ok = DelayRunnerClass.save_bar(bars_dict, out_path)
            return bool(ok)
        except Exception:
            return False
    return False
