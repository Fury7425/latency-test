# app/ui/app.py
import os, sys
import customtkinter as ctk
from tkinter import PhotoImage

from .theme import apply_theme
from ..config import load_config
from ..core.audio import AudioCore
from .pages.latency_page import LatencyPage
from .pages.lab_page import LabPage
from .pages.devices_page import DevicesPage
from .pages.results_page import ResultsPage
from .glass import Background, GlassCard, resource_path

class MainApp(ctk.CTk):
    def __init__(self):
        self.cfg = load_config()
        apply_theme(self.cfg["ui"])
        super().__init__()

        # Window basics
        self.title("Pawdio-Lab")
        self.geometry("1200x800")
        self.minsize(1060, 720)

        # Icons (optional if you already set elsewhere)
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

        # Background
        bg_path = resource_path(os.path.join("app", "assets", "bg.jpg"))
        self.bg = Background(self, image_path=bg_path)
        self.bg.place(relx=0, rely=0, relwidth=1, relheight=1)

        # 2-column layout: sidebar (glass) + main (transparent)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar as glass card
        self.sidebar_card = GlassCard(self)
        self.sidebar_card.grid(row=0, column=0, sticky="nsw", padx=16, pady=16)
        self.sidebar = self.sidebar_card.inner
        self.sidebar.configure(corner_radius=18)
        self.sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text="Pawdio‑Lab",
            font=ctk.CTkFont(size=22, weight="bold")
        ).grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.btn_latency = ctk.CTkButton(self.sidebar, text="Latency", command=lambda: self._show("latency"))
        self.btn_latency.grid(row=2, column=0, padx=14, pady=6, sticky="ew")

        self.btn_devices = ctk.CTkButton(self.sidebar, text="Devices / Settings", command=lambda: self._show("devices"))
        self.btn_devices.grid(row=3, column=0, padx=14, pady=6, sticky="ew")

        self.btn_results = ctk.CTkButton(self.sidebar, text="Results / Export", command=lambda: self._show("results"))
        self.btn_results.grid(row=4, column=0, padx=14, pady=6, sticky="ew")

        self.btn_lab = None

        # Main container
        self.main_card = GlassCard(self)
        self.main_card.grid(row=0, column=1, sticky="nsew", padx=(8,16), pady=16)
        self.main = self.main_card.inner
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # Audio core
        self.core = AudioCore(
            sample_rate=self.cfg["last_settings"]["sample_rate"],
            chunk_size=1024,
            duration=self.cfg["last_settings"]["duration"],
            output_device_index=self.cfg["last_settings"]["output_device_index"],
            input_device_index=self.cfg["last_settings"]["input_device_index"]
        )

        # Pages (unchanged)
        self.pages = {}
        self.pages["latency"] = LatencyPage(self.main, self.core, self.cfg, self._log)
        self.pages["devices"] = DevicesPage(self.main, self.core, self.cfg, self._log, on_toggle_labs=self._toggle_labs)
        self.pages["results"] = ResultsPage(self.main, self._log_sink)
        if self.cfg["ui"].get("labs_enabled", True):
            self._add_labs_page()
        self._show("latency")

    def _add_labs_page(self):
        self.pages["lab"] = LabPage(self.main, self.core, self.cfg, self._log)
        if self.btn_lab is None:
            self.btn_lab = ctk.CTkButton(self.sidebar, text="Lab Tests", command=lambda: self._show("lab"))
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

    def _show(self, key):
        for _, p in self.pages.items():
            p.grid_remove()
        self.pages[key].grid(row=0, column=0, sticky="nsew")

    def _log(self, msg):
        if "results" in self.pages:
            self.pages["results"].append(msg)

    def _log_sink(self, msg):
        pass
