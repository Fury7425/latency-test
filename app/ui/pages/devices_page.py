# app/ui/pages/devices_page.py
import customtkinter as ctk

# Robust imports
try:
    from ...config import save_config
    from ..glass import GlassCard
except ImportError:
    from config import save_config
    from app.ui.glass import GlassCard

# PyAudio fallback for enumeration
try:
    import pyaudio
except Exception:
    pyaudio = None


class DevicesPage(ctk.CTkFrame):
    def __init__(self, master, core, cfg, log_fn, on_toggle_labs=None):
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

        # Devices card
        dev = GlassCard(self); dev.grid(row=1, column=0, padx=18, pady=8, sticky="ew")
        d = dev.inner
        for i in range(2): d.grid_columnconfigure(i, weight=1)

        outs = self._list_output_devices(core)
        ins  = self._list_input_devices(core)
        if not outs: outs = [(0, "Default Output")]
        if not ins:  ins  = [(0, "Default Input")]

        # read saved indexes safely
        out_idx_saved = self.cfg["last_settings"].get("output_device_index", None)
        in_idx_saved  = self.cfg["last_settings"].get("input_device_index", None)

        out_idx_default = outs[0][0]
        in_idx_default  = ins[0][0]

        out_idx = out_idx_default if not self._is_int(out_idx_saved) else int(out_idx_saved)
        in_idx  = in_idx_default  if not self._is_int(in_idx_saved)  else int(in_idx_saved)

        self.var_out = ctk.StringVar(value=str(out_idx))
        self.var_in  = ctk.StringVar(value=str(in_idx))

        ctk.CTkLabel(d, text="Output Device").grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")
        self.menu_out = ctk.CTkOptionMenu(
            d,
            values=[f"{i}: {name}" for i, name in outs],
            command=lambda *_: self._apply_devices()
        )
        self._set_menu_from_index(self.menu_out, outs, out_idx)
        self.menu_out.grid(row=1, column=0, padx=14, pady=(0, 12), sticky="ew")

        ctk.CTkLabel(d, text="Input Device").grid(row=0, column=1, padx=14, pady=(12, 6), sticky="w")
        self.menu_in = ctk.CTkOptionMenu(
            d,
            values=[f"{i}: {name}" for i, name in ins],
            command=lambda *_: self._apply_devices()
        )
        self._set_menu_from_index(self.menu_in, ins, in_idx)
        self.menu_in.grid(row=1, column=1, padx=14, pady=(0, 12), sticky="ew")

        # Audio settings
        s = GlassCard(self); s.grid(row=2, column=0, padx=18, pady=8, sticky="ew")
        sett = s.inner
        for i in range(6): sett.grid_columnconfigure(i, weight=1)

        sr_saved  = self.cfg["last_settings"].get("sample_rate", 48000)
        dur_saved = self.cfg["last_settings"].get("duration", 3.0)
        self.var_sr  = ctk.StringVar(value=str(sr_saved))
        self.var_dur = ctk.StringVar(value=str(dur_saved))

        ctk.CTkLabel(sett, text="Sample Rate (Hz)").grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")
        ctk.CTkEntry(sett, textvariable=self.var_sr, width=120).grid(row=0, column=1, padx=(8, 14), pady=(12,6), sticky="w")

        ctk.CTkLabel(sett, text="Capture Duration (s)").grid(row=0, column=2, padx=14, pady=(12, 6), sticky="w")
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

    # ---------- enumeration helpers ----------
    def _list_output_devices(self, core):
        if hasattr(core, "list_output_devices"):
            try: return list(core.list_output_devices())
            except Exception: pass
        return self._pyaudio_enum(kind="output")

    def _list_input_devices(self, core):
        if hasattr(core, "list_input_devices"):
            try: return list(core.list_input_devices())
            except Exception: pass
        return self._pyaudio_enum(kind="input")

    def _pyaudio_enum(self, kind="output"):
        devs = []
        if pyaudio is None: return devs
        try:
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                name = info.get("name", f"Device {i}")
                if kind == "output" and int(info.get("maxOutputChannels", 0)) > 0:
                    devs.append((i, name))
                if kind == "input" and int(info.get("maxInputChannels", 0)) > 0:
                    devs.append((i, name))
            pa.terminate()
        except Exception:
            pass
        return devs

    # ---------- apply handlers ----------
    def _is_int(self, v):
        try:
            int(v)
            return True
        except Exception:
            return False

    def _set_menu_from_index(self, menu: ctk.CTkOptionMenu, items, idx):
        chosen = None
        for i, name in items:
            if i == idx:
                chosen = f"{i}: {name}"
                break
        if chosen is None and items:
            chosen = f"{items[0][0]}: {items[0][1]}"
        if chosen:
            menu.set(chosen)

    def _apply_devices(self):
        try:
            out_idx = int(self.menu_out.get().split(":")[0])
            in_idx  = int(self.menu_in.get().split(":")[0])
        except Exception:
            return

        self.cfg["last_settings"]["output_device_index"] = out_idx
        self.cfg["last_settings"]["input_device_index"]  = in_idx
        save_config(self.cfg)

        if hasattr(self.core, "set_devices"):
            try:
                self.core.set_devices(output_device_index=out_idx, input_device_index=in_idx)
            except Exception as e:
                self.log(f"[DEV] set_devices failed: {e}")
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

        if hasattr(self.core, "set_timing"):
            try:
                self.core.set_timing(sample_rate=sr, duration=dur)
            except Exception as e:
                self.log(f"[AUDIO] set_timing failed: {e}")
        else:
            try:
                self.core.sample_rate = sr
                self.core.duration = dur
            except Exception:
                pass
        self.log(f"[AUDIO] SR={sr}, DUR={dur}s")

    def _toggle_labs(self):
        enabled = bool(self.var_labs.get())
        self.cfg["ui"]["labs_enabled"] = enabled
        save_config(self.cfg)
        self.on_toggle_labs(enabled)
        self.log(f"[UI] Labs enabled={enabled}")
