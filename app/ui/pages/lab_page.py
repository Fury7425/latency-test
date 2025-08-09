# app/ui/pages/lab_page.py
import os
import json
import customtkinter as ctk

from ...tests import balance, crosstalk, sweep_fr, thd, isolation
from ...core.utils import ensure_dir


class LabPage(ctk.CTkFrame):
    def __init__(self, master, core, cfg, log_fn):
        super().__init__(master, corner_radius=0)
        self.core = core
        self.cfg = cfg
        self.log = log_fn
        self.results = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self, text="Lab Tests", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")

        # -----------------------
        # Channel Balance
        # -----------------------
        bal = ctk.CTkFrame(self)
        bal.grid(row=1, column=0, padx=18, pady=12, sticky="nsew")
        self.var_bal = ctk.StringVar(value="1000")
        ctk.CTkLabel(bal, text="Channel Balance").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(bal, text="Freq (Hz)").grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkEntry(bal, textvariable=self.var_bal, width=90).grid(
            row=1, column=1, sticky="w", padx=8
        )
        ctk.CTkButton(bal, text="Run", command=self.on_balance).grid(
            row=2, column=0, padx=8, pady=8, sticky="w"
        )

        # -----------------------
        # Crosstalk
        # -----------------------
        xt = ctk.CTkFrame(self)
        xt.grid(row=1, column=1, padx=18, pady=12, sticky="nsew")
        self.var_xt = ctk.StringVar(value="1000")
        self.var_dir = ctk.StringVar(value="LtoR")
        ctk.CTkLabel(xt, text="Crosstalk").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(xt, text="Freq (Hz)").grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkEntry(xt, textvariable=self.var_xt, width=90).grid(
            row=1, column=1, sticky="w", padx=8
        )
        ctk.CTkOptionMenu(xt, values=["LtoR", "RtoL"], variable=self.var_dir).grid(
            row=1, column=2, padx=8
        )
        ctk.CTkButton(xt, text="Run", command=self.on_crosstalk).grid(
            row=2, column=0, padx=8, pady=8, sticky="w"
        )

        # -----------------------
        # Sweep FR (multi-run overlay + average)
        # -----------------------
        fr = ctk.CTkFrame(self)
        fr.grid(row=2, column=0, padx=18, pady=12, sticky="nsew")

        self.var_dur = ctk.StringVar(value="6.0")
        self.var_repeats = ctk.StringVar(value="3")  # NEW: number of runs

        ctk.CTkLabel(fr, text="Sweep FR (relative)").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(fr, text="Duration (s)").grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkEntry(fr, textvariable=self.var_dur, width=90).grid(
            row=1, column=1, sticky="w", padx=8
        )

        ctk.CTkLabel(fr, text="Repeats").grid(row=1, column=2, sticky="w", padx=8)  # NEW
        ctk.CTkEntry(fr, textvariable=self.var_repeats, width=90).grid(              # NEW
            row=1, column=3, sticky="w", padx=8
        )

        ctk.CTkButton(fr, text="Run", command=self.on_sweep).grid(
            row=2, column=0, padx=8, pady=8, sticky="w"
        )

        # -----------------------
        # THD
        # -----------------------
        thd_card = ctk.CTkFrame(self)
        thd_card.grid(row=2, column=1, padx=18, pady=12, sticky="nsew")
        ctk.CTkLabel(thd_card, text="THD (100/1k/6k Hz)").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ctk.CTkButton(thd_card, text="Run", command=self.on_thd).grid(
            row=1, column=0, padx=8, pady=8, sticky="w"
        )

        # -----------------------
        # Isolation
        # -----------------------
        iso = ctk.CTkFrame(self)
        iso.grid(row=3, column=0, padx=18, pady=12, sticky="nsew")
        ctk.CTkLabel(iso, text="Isolation (Inside → Outside)").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ctk.CTkButton(iso, text="Run", command=self.on_isolation).grid(
            row=1, column=0, padx=8, pady=8, sticky="w"
        )

        # -----------------------
        # Results + Export
        # -----------------------
        res = ctk.CTkFrame(self)
        res.grid(row=3, column=1, padx=18, pady=12, sticky="nsew")
        res.grid_rowconfigure(1, weight=1)
        res.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(res, text="Lab Results").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        self.box = ctk.CTkTextbox(res)
        self.box.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        row = ctk.CTkFrame(res)
        row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkButton(row, text="Export LAST (JSON)", command=self.export_last).pack(
            side="left", padx=4
        )
        ctk.CTkButton(row, text="Export ALL (JSON)", command=self.export_all).pack(
            side="left", padx=4
        )

    # -----------------------
    # Handlers
    # -----------------------
    def on_balance(self):
        r = balance.run(self.core, self.log, freq=float(self.var_bal.get()))
        self._push(r)

    def on_crosstalk(self):
        r = crosstalk.run(
            self.core, self.log, freq=float(self.var_xt.get()), direction=self.var_dir.get()
        )
        self._push(r)

    def on_sweep(self):
        out = self.cfg["last_settings"].get("output_dir", os.getcwd())
        r = sweep_fr.run(
            self.core,
            self.log,
            duration=float(self.var_dur.get()),
            repeats=int(self.var_repeats.get()),  # pass repeats to multi-run sweep
            save_plot_dir=out,
        )
        self._push(r)

    def on_thd(self):
        r = thd.run(self.core, self.log)
        self._push(r)

    def on_isolation(self):
        r = isolation.run(self.core, self.log)
        self._push(r)

    # -----------------------
    # Result helpers
    # -----------------------
    def _push(self, res):
        self.results.append(res.to_dict())
        self.box.insert("end", json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n\n")
        self.box.see("end")

    def export_last(self):
        if not self.results:
            return
        out_dir = self.cfg["last_settings"].get("output_dir", os.getcwd())
        ensure_dir(out_dir)
        last = self.results[-1]
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"lab_last_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(last, f, indent=2, ensure_ascii=False)
        self.log(f"[EXPORT] last -> {path}")

    def export_all(self):
        if not self.results:
            return
        out_dir = self.cfg["last_settings"].get("output_dir", os.getcwd())
        ensure_dir(out_dir)
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"lab_all_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        self.log(f"[EXPORT] all -> {path}")
