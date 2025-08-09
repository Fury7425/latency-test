# app/ui/pages/devices_page.py
import os
import customtkinter as ctk

# Robust imports
try:
    from ...core.audio import AudioCore
    from ...config import save_config
    from ..glass import GlassCard
except ImportError:
    from core.audio import AudioCore
    from config import save_config
    from app.ui.glass import GlassCard


class DevicesPage(ctk.CTkFrame):
    """
    Devices / Settings page (glassmorphism)
    - Input/Output device selectors
    - Sample rate & duration
    - Labs toggle (shows/hides Lab page)
    """
    def __init__(self, master, core: "AudioCore", cfg, log_fn, on_toggle_labs=None):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self.core = core
        self.cfg = cfg
        self.log = log_fn
        self.on_toggle_labs = on_toggle_labs or (lambda _enabled: None)

        self.grid_columnconfigure(0, weight=1)

        # Header
        hdr = GlassCard(self); hdr.grid(row=0, column=0, padx=18, pady=(16, 8), sticky="ew")
        h = hdr.inner
        ctk.CTkLabel(h, text="Devices & Settings", font=ctk.CTkFont(size=20, weight="bold"))\
            .grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")
        ctk.CTkLabel(h, text="Select input / output, adjust sample rate and capture duration.")\
            .grid(row=1, column=0, padx=14, pady=(0, 12), sticky="w")

        # Devices
        dev = GlassCard(self); dev.grid(row=1, column=0, padx=18, pady=8, sticky="ew")
        d = dev.inner
        for i in range(2): d.grid_columnconfigure(i, weight=1)

        # Build device lists
        outs = self.core.list_output_devices()
        ins  = self.core.list_input_devices()

        self.var_out = ctk.StringVar(value=str(self.cfg["last_settings"].get("output_device_index", "")))
        self.var_in  = ctk.StringVar(value=str(self.cfg["last_settings"].get("input_device_index", "")))

        ctk.CTkLabel(d, text="Output Device").grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")
        self.menu_out = ctk.CTkOptionMenu(
            d,
            values=[f"{i}: {name}" for i, name in outs],
            command=lambda *_: self._apply_devices(),
        )
        self._set_menu_from_index(self.menu_out, outs, self.var_out.get())
        self.menu_out.grid(row=1, column=0, padx=14, pady=(0, 12), sticky="ew")

        ctk.CTkLabel(d, text="Input Device").grid(row=0, column=1, padx=14, pady=(12, 6), sticky="w")
        self.menu_in = ctk.CTkOptionMenu(
            d,
            values=[f"{i}: {name}" for i, name in ins],
            command=lambda *_: self._apply_devices(),
        )
        self._set_menu_from_index(self.menu_in, ins, self.var_in.get())
        self.menu_in.grid(row=1, column=1, padx=14, pady=(0, 12), sticky="ew")

        # Audio settings
        s = GlassCard(self); s.grid(row=2, column=0, padx=18, pady=8, sticky="ew")
        sett = s.inner
        for i in range(6): sett.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(sett, text="Sample Rate (Hz)").grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")
        self.var_sr = ctk.StringVar(value=str(self.cfg["last_settings"].get("sample_rate", 48000)))
        ctk.CTkEntry(sett, textvariable=self.var_sr, width=120).grid(row=0, column=1, padx=(8, 14), pady=(12,6), sticky="w")

        ctk.CTkLabel(sett, text="Capture Duration (s)").grid(row=0, column=2, padx=14, pady=(12, 6), sticky="w")
        self.var_dur = ctk.StringVar(value=str(self.cfg["last_settings"].get("duration", 3.0)))
        ctk.CTkEntry(sett, textvariable=self.var_dur, width=120).grid(row=0, column=3, padx=(8,14), pady=(12,6), sticky="w")

        ctk.CTkButton(sett, text="Apply", command=self._apply_audio)\
           .grid(row=0, column=5, padx=14, pady=(12,6), sticky="e")

        # Labs toggle
        labs = GlassCard(self); labs.grid(row=3, column=0, padx=18, pady=(8,18), sticky="ew")
        l = labs.inner
        l.grid_columnconfigure(0, weight=1)
        self.var_labs = ctk.BooleanVar(value=bool(self.cfg["ui"].get("labs_enabled", True)))
        ctk.CTkSwitch(l, text="Enable Lab Tests", variable=self.var_labs, command=self._toggle_labs)\
            .grid(row=0, column=0, padx=14, pady=(12, 12), sticky="w")

    # helpers
    def _set_menu_from_index(self, menu: ctk.CTkOptionMenu, items, idx_str):
        try:
            idx = int(idx_str)
        except Exception:
            idx = items[0][0] if items else 0
        for i, name in items:
            if i == idx:
                menu.set(f"{i}: {name}")
                return
        if items:
            menu.set(f"{items[0][0]}: {items[0][1]}")

    def _apply_devices(self):
        out_txt = self.menu_out.get()
        in_txt  = self.menu_in.get()
        try:
            out_idx = int(out_txt.split(":")[0])
            in_idx  = int(in_txt.split(":")[0])
        except Exception:
            return
        self.cfg["last_settings"]["output_device_index"] = out_idx
        self.cfg["last_settings"]["input_device_index"]  = in_idx
        save_config(self.cfg)
        self.core.set_devices(output_device_index=out_idx, input_device_index=in_idx)
        self.log(f"[DEV] Output={out_idx}, Input={in_idx}")

    def _apply_audio(self):
        try:
            sr = int(float(self.var_sr.get()))
            dur = float(self.var_dur.get())
        except Exception:
            self.log("[WARN] invalid sample rate or duration")
            return
        self.cfg["last_settings"]["sample_rate"] = sr
        self.cfg["last_settings"]["duration"] = dur
        save_config(self.cfg)
        self.core.set_timing(sample_rate=sr, duration=dur)
        self.log(f"[AUDIO] SR={sr}, DUR={dur}s")

    def _toggle_labs(self):
        enabled = bool(self.var_labs.get())
        self.cfg["ui"]["labs_enabled"] = enabled
        save_config(self.cfg)
        self.on_toggle_labs(enabled)
        self.log(f"[UI] Labs enabled={enabled}")
