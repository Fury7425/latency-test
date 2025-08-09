# app/ui/pages/lab_page.py
import os
import json
import customtkinter as ctk

# Robust imports
try:
    from ...tests import balance, crosstalk, sweep_fr, thd, isolation
    from ...core.utils import ensure_dir
    from ..glass import GlassCard
except ImportError:
    from tests import balance, crosstalk, sweep_fr, thd, isolation
    from core.utils import ensure_dir
    from app.ui.glass import GlassCard


class LabPage(ctk.CTkFrame):
    def __init__(self, master, core, cfg, log_fn):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self.core = core
        self.cfg = cfg
        self.log = log_fn
        self.results = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Title card
        title = GlassCard(self); title.grid(row=0, column=0, columnspan=2, padx=18, pady=(16,8), sticky="ew")
        t = title.inner
        ctk.CTkLabel(t, text="Lab Tests", font=ctk.CTkFont(size=20, weight="bold"))\
            .grid(row=0, column=0, padx=14, pady=(12,6), sticky="w")

        # Balance
        bal = GlassCard(self); bal.grid(row=1, column=0, padx=18, pady=8, sticky="nsew")
        b = bal.inner
        self.var_bal = ctk.StringVar(value="1000")
        ctk.CTkLabel(b, text="Channel Balance", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=8, pady=(10,4), sticky="w")
        ctk.CTkLabel(b, text="Freq (Hz)").grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkEntry(b, textvariable=self.var_bal, width=100).grid(row=1, column=1, sticky="w", padx=8)
        ctk.CTkButton(b, text="Run", command=self.on_balance).grid(row=2, column=0, padx=8, pady=(8,10), sticky="w")

        # Crosstalk
        xt = GlassCard(self); xt.grid(row=1, column=1, padx=18, pady=8, sticky="nsew")
        x = xt.inner
        self.var_xt = ctk.StringVar(value="1000")
        self.var_dir = ctk.StringVar(value="LtoR")
        ctk.CTkLabel(x, text="Crosstalk", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=8, pady=(10,4), sticky="w")
        ctk.CTkLabel(x, text="Freq (Hz)").grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkEntry(x, textvariable=self.var_xt, width=100).grid(row=1, column=1, sticky="w", padx=8)
        ctk.CTkOptionMenu(x, values=["LtoR", "RtoL"], variable=self.var_dir).grid(row=1, column=2, padx=8)
        ctk.CTkButton(x, text="Run", command=self.on_crosstalk).grid(row=2, column=0, padx=8, pady=(8,10), sticky="w")

        # Sweep FR (multi-run)
        fr = GlassCard(self); fr.grid(row=2, column=0, padx=18, pady=8, sticky="nsew")
        f = fr.inner
        self.var_dur = ctk.StringVar(value="6.0")
        self.var_repeats = ctk.StringVar(value="3")
        ctk.CTkLabel(f, text="Sweep FR (relative)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=8, pady=(10,4), sticky="w")
        ctk.CTkLabel(f, text="Duration (s)").grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkEntry(f, textvariable=self.var_dur, width=100).grid(row=1, column=1, sticky="w", padx=8)
        ctk.CTkLabel(f, text="Repeats").grid(row=1, column=2, sticky="w", padx=8)
        ctk.CTkEntry(f, textvariable=self.var_repeats, width=100).grid(row=1, column=3, sticky="w", padx=8)
        ctk.CTkButton(f, text="Run", command=self.on_sweep).grid(row=2, column=0, padx=8, pady=(8,10), sticky="w")

        # THD
        thd_card = GlassCard(self); thd_card.grid(row=2, column=1, padx=18, pady=8, sticky="nsew")
        th = thd_card.inner
        ctk.CTkLabel(th, text="THD (100 / 1k / 6k Hz)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=8, pady=(10,4), sticky="w")
        ctk.CTkButton(th, text="Run", command=self.on_thd).grid(row=1, column=0, padx=8, pady=(8,10), sticky="w")

        # Isolation
        iso = GlassCard(self); iso.grid(row=3, column=0, padx=18, pady=8, sticky="nsew")
        ii = iso.inner
        ctk.CTkLabel(ii, text="Isolation (Inside → Outside)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=8, pady=(10,4), sticky="w")
        ctk.CTkButton(ii, text="Run", command=self.on_isolation).grid(row=1, column=0, padx=8, pady=(8,10), sticky="w")

        # Results + Export
        res = GlassCard(self); res.grid(row=3, column=1, padx=18, pady=8, sticky="nsew")
        rr = res.inner
        rr.grid_rowconfigure(1, weight=1)
        rr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(rr, text="Lab Results", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=8, pady=(10,4), sticky="w")
        self.box = ctk.CTkTextbox(rr)
        self.box.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        row = ctk.CTkFrame(rr, fg_color="transparent"); row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkButton(row, text="Export LAST (JSON)", command=self.export_last).pack(side="left", padx=4)
        ctk.CTkButton(row, text="Export ALL (JSON)", command=self.export_all).pack(side="left", padx=4)

    # Handlers
    def on_balance(self):
        r = balance.run(self.core, self.log, freq=float(self.var_bal.get()))
        self._push(r)

    def on_crosstalk(self):
        r = crosstalk.run(self.core, self.log, freq=float(self.var_xt.get()), direction=self.var_dir.get())
        self._push(r)

    def on_sweep(self):
        out = self.cfg["last_settings"].get("output_dir", os.getcwd())
        r = sweep_fr.run(
            self.core, self.log,
            duration=float(self.var_dur.get()),
            repeats=int(self.var_repeats.get()),
            save_plot_dir=out
        )
        self._push(r)

    def on_thd(self):
        r = thd.run(self.core, self.log)
        self._push(r)

    def on_isolation(self):
        r = isolation.run(self.core, self.log)
        self._push(r)

    # Result helpers
    def _push(self, res):
        self.results.append(res.to_dict())
        self.box.insert("end", json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n\n")
        self.box.see("end")

    def export_last(self):
        if not self.results: return
        out_dir = self.cfg["last_settings"].get("output_dir", os.getcwd())
        ensure_dir(out_dir)
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"lab_last_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results[-1], f, indent=2, ensure_ascii=False)
        self.log(f"[EXPORT] last -> {path}")

    def export_all(self):
        if not self.results: return
        out_dir = self.cfg["last_settings"].get("output_dir", os.getcwd())
        ensure_dir(out_dir)
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"lab_all_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        self.log(f"[EXPORT] all -> {path}")
