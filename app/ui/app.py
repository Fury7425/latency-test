# app/ui/app.py
import os
import sys
import customtkinter as ctk
from tkinter import PhotoImage

# Robust imports: package first, then flat
try:
    from .theme import apply_theme
    from ..config import load_config
    from ..core.audio import AudioCore
    from .pages.latency_page import LatencyPage
    from .pages.lab_page import LabPage
    from .pages.devices_page import DevicesPage
    from .pages.results_page import ResultsPage
    from .glass import Background, GlassCard, resource_path
except Exception:
    from ui.theme import apply_theme
    from config import load_config
    from core.audio import AudioCore
    from ui.pages.latency_page import LatencyPage
    from ui.pages.lab_page import LabPage
    from ui.pages.devices_page import DevicesPage
    from ui.pages.results_page import ResultsPage
    from ui.glass import Background, GlassCard, resource_path


class MainApp(ctk.CTk):
    def __init__(self):
        # Config + theme
        self.cfg = load_config()
        apply_theme(self.cfg.get("ui", {}))

        super().__init__()

        # Window basics
        self.title("Pawdio-Lab")
        self.geometry("1200x800")
        self.minsize(1060, 720)

        # Icons (PNG for window; ICO for Windows taskbar/EXE)
        try:
            png_path = resource_path("paw.png")
            if os.path.exists(png_path):
                self.iconphoto(True, PhotoImage(file=png_path))
        except Exception:
            pass

        try:
            ico_path = resource_path("paw.ico")
            if os.path.exists(ico_path):
                self.iconbitmap(ico_path)
        except Exception:
            pass

        # Background (blurred image)
        bg_path = resource_path(os.path.join("app", "assets", "bg.jpg"))
        self.bg = Background(self, image_path=bg_path)
        self.bg.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Layout: sidebar + main (both on glass cards)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar (glass)
        self.sidebar_card = GlassCard(self)
        self.sidebar_card.grid(row=0, column=0, sticky="nsw", padx=16, pady=16)
        self.sidebar = self.sidebar_card.inner
        self.sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            self.sidebar, text="Pawdio‑Lab",
            font=ctk.CTkFont(size=22, weight="bold")
        ).grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.btn_latency = ctk.CTkButton(
            self.sidebar, text="Latency",
            command=lambda: self._show("latency")
        )
        self.btn_latency.grid(row=2, column=0, padx=14, pady=6, sticky="ew")

        self.btn_devices = ctk.CTkButton(
            self.sidebar, text="Devices / Settings",
            command=lambda: self._show("devices")
        )
        self.btn_devices.grid(row=3, column=0, padx=14, pady=6, sticky="ew")

        self.btn_results = ctk.CTkButton(
            self.sidebar, text="Results / Export",
            command=lambda: self._show("results")
        )
        self.btn_results.grid(row=4, column=0, padx=14, pady=6, sticky="ew")

        self.btn_lab = None  # created lazily if labs are enabled

        # Main container (glass)
        self.main_card = GlassCard(self)
        self.main_card.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
        self.main = self.main_card.inner
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # Audio core
        ls = self.cfg.get("last_settings", {})
        self.core = AudioCore(
            sample_rate=ls.get("sample_rate", 48000),
            chunk_size=1024,
            duration=ls.get("duration", 3.0),
            output_device_index=ls.get("output_device_index", None),
            input_device_index=ls.get("input_device_index", None),
        )

        # Pages
        self.pages = {}
        self.pages["latency"] = LatencyPage(self.main, self.core, self.cfg, self._log)
        self.pages["devices"] = DevicesPage(
            self.main, self.core, self.cfg, self._log, on_toggle_labs=self._toggle_labs
        )
        self.pages["results"] = ResultsPage(self.main, self._log_sink)

        if self.cfg.get("ui", {}).get("labs_enabled", True):
            self._add_labs_page()

        # Default page
        self._show("latency")

    # ---------- page wiring ----------
    def _add_labs_page(self):
        self.pages["lab"] = LabPage(self.main, self.core, self.cfg, self._log)
        if self.btn_lab is None:
            self.btn_lab = ctk.CTkButton(
                self.sidebar, text="Lab Tests",
                command=lambda: self._show("lab")
            )
            self.btn_lab.grid(row=5, column=0, padx=14, pady=6, sticky="ew")

    def _remove_labs_page(self):
        if "lab" in self.pages:
            self.pages["lab"].grid_forget()
            del self.pages["lab"]
        if self.btn_lab is not None:
            self.btn_lab.destroy()
            self.btn_lab = None

    def _toggle_labs(self, enabled: bool):
        if enabled:
            self._add_labs_page()
        else:
            self._remove_labs_page()

    def _show(self, key: str):
        for k, p in self.pages.items():
            p.grid_remove()
        self.pages[key].grid(row=0, column=0, sticky="nsew")

    # ---------- logging sinks ----------
    def _log(self, msg: str):
        if "results" in self.pages:
            self.pages["results"].append(msg)

    def _log_sink(self, msg: str):
        # Placeholder if you ever want to hook external logs
        pass
