
import customtkinter as ctk
from ...config import save_config

class DevicesPage(ctk.CTkFrame):
    def __init__(self, master, core, cfg, log_fn, on_toggle_labs):
        super().__init__(master, corner_radius=0)
        self.core = core; self.cfg = cfg; self.log = log_fn; self.on_toggle_labs = on_toggle_labs
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text="Devices / Settings", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, columnspan=2, padx=18, pady=(18,4), sticky="w")

        card = ctk.CTkFrame(self); card.grid(row=1, column=0, columnspan=2, padx=18, pady=12, sticky="ew")
        for i in range(6): card.grid_columnconfigure(i, weight=1)
        ctk.CTkLabel(card, text="Sample Rate").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.sr_var = ctk.IntVar(value=self.cfg["last_settings"]["sample_rate"])
        ctk.CTkEntry(card, textvariable=self.sr_var, width=120).grid(row=0, column=1, padx=8, pady=8, sticky="w")
        ctk.CTkLabel(card, text="Signal Duration (s)").grid(row=0, column=2, padx=8, pady=8, sticky="w")
        self.dur_var = ctk.DoubleVar(value=self.cfg["last_settings"]["duration"])
        ctk.CTkEntry(card, textvariable=self.dur_var, width=120).grid(row=0, column=3, padx=8, pady=8, sticky="w")
        ctk.CTkButton(card, text="Apply", command=self._apply).grid(row=0, column=5, padx=8, pady=8, sticky="e")

        dev = ctk.CTkFrame(self); dev.grid(row=2, column=0, columnspan=2, padx=18, pady=12, sticky="ew")
        dev.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(dev, text="Output Device").grid(row=0, column=0, padx=8, pady=(8,4), sticky="w")
        self.out_menu = ctk.CTkOptionMenu(dev, values=["<refresh>"]); self.out_menu.grid(row=0, column=1, padx=8, pady=(8,4), sticky="ew")
        ctk.CTkLabel(dev, text="Input Device").grid(row=1, column=0, padx=8, pady=(4,8), sticky="w")
        self.in_menu = ctk.CTkOptionMenu(dev, values=["<refresh>"]); self.in_menu.grid(row=1, column=1, padx=8, pady=(4,8), sticky="ew")
        ctk.CTkButton(dev, text="Refresh Devices", command=self._refresh_devices).grid(row=0, column=2, rowspan=2, padx=8)

        ap = ctk.CTkFrame(self); ap.grid(row=3, column=0, columnspan=2, padx=18, pady=12, sticky="ew")
        ap.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(ap, text="Appearance").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.appearance = ctk.CTkOptionMenu(ap, values=["Light","Dark","System"]); self.appearance.set(self.cfg["ui"].get("appearance_mode", "Dark")); self.appearance.grid(row=0, column=1, padx=8, sticky="w")
        ctk.CTkLabel(ap, text="Accent").grid(row=0, column=2, padx=8, pady=8, sticky="w")
        self.color = ctk.CTkOptionMenu(ap, values=["blue","dark-blue","green"]); self.color.set(self.cfg["ui"].get("color_theme","blue")); self.color.grid(row=0, column=3, padx=8, sticky="w")

        self.labs_var = ctk.BooleanVar(value=bool(self.cfg["ui"].get("labs_enabled", True)))
        ctk.CTkSwitch(ap, text="Enable Lab Tests (experimental)", variable=self.labs_var, command=self._toggle_labs).grid(row=1, column=0, columnspan=2, padx=8, pady=(8,6), sticky="w")

        self._refresh_devices()

    def _apply(self):
        try:
            self.core.sample_rate = int(self.sr_var.get())
            self.core.duration = float(self.dur_var.get())
            self.cfg["last_settings"]["sample_rate"] = self.core.sample_rate
            self.cfg["last_settings"]["duration"] = self.core.duration
            self.cfg["ui"]["appearance_mode"] = self.appearance.get()
            self.cfg["ui"]["color_theme"] = self.color.get()
            save_config(self.cfg)
            self.log(f"[SET] sr={self.core.sample_rate}, dur={self.core.duration}s, theme={self.appearance.get()}/{self.color.get()}")
        except Exception as e:
            self.log(f"[ERROR] settings: {e}")

    def _refresh_devices(self):
        devs = self.core.list_devices()
        out_values, in_values = [], []
        self._out_idx = {}; self._in_idx = {}
        for i, info in enumerate(devs):
            name = info.get("name","?"); m_in = int(info.get("maxInputChannels",0)); m_out=int(info.get("maxOutputChannels",0)); rate = int(info.get("defaultSampleRate",0))
            label = f"[{i}] {name} (in:{m_in}, out:{m_out}, rate:{rate})"
            if m_out>0: out_values.append(label); self._out_idx[label]=i
            if m_in>0: in_values.append(label); self._in_idx[label]=i
        self.out_menu.configure(values=out_values or ["<no outputs>"]); self.in_menu.configure(values=in_values or ["<no inputs>"])
        out_stored = self.cfg["last_settings"].get("output_device_index")
        in_stored = self.cfg["last_settings"].get("input_device_index")
        if out_values:
            target = next((lbl for lbl,idx in self._out_idx.items() if idx==out_stored), out_values[0]); self.out_menu.set(target)
        if in_values:
            target = next((lbl for lbl,idx in self._in_idx.items() if idx==in_stored), in_values[0]); self.in_menu.set(target)

    def use_selected_devices(self):
        out_idx = self._out_idx.get(self.out_menu.get()); in_idx = self._in_idx.get(self.in_menu.get())
        if out_idx is None or in_idx is None: return
        self.core.set_devices(output_index=out_idx, input_index=in_idx)
        self.cfg["last_settings"]["output_device_index"] = out_idx
        self.cfg["last_settings"]["input_device_index"] = in_idx
        save_config(self.cfg); self.log(f"[SET] devices out={out_idx}, in={in_idx}")

    def _toggle_labs(self):
        self.cfg["ui"]["labs_enabled"] = bool(self.labs_var.get()); save_config(self.cfg)
        self.on_toggle_labs(bool(self.labs_var.get()))
