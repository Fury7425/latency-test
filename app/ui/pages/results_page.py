# app/ui/pages/results_page.py
import os
import json
import customtkinter as ctk

# Robust imports
try:
    from ..glass import GlassCard
    from ...core.utils import ensure_dir
except ImportError:
    from app.ui.glass import GlassCard
    from core.utils import ensure_dir


class ResultsPage(ctk.CTkFrame):
    """
    Results / Export (glassmorphism)
    - Live log sink
    - Export log to TXT
    - Clear
    """
    def __init__(self, master, log_sink):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self._sink = log_sink
        self.messages = []

        self.grid_columnconfigure(0, weight=1)

        # Header
        hdr = GlassCard(self); hdr.grid(row=0, column=0, padx=18, pady=(16,8), sticky="ew")
        h = hdr.inner
        ctk.CTkLabel(h, text="Results & Export", font=ctk.CTkFont(size=20, weight="bold"))\
            .grid(row=0, column=0, padx=14, pady=(12,6), sticky="w")
        ctk.CTkLabel(h, text="Run logs and saved file paths will appear here.")\
            .grid(row=1, column=0, padx=14, pady=(0,12), sticky="w")

        # Log box
        card = GlassCard(self); card.grid(row=1, column=0, padx=18, pady=8, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)
        c = card.inner
        c.grid_rowconfigure(0, weight=1)
        c.grid_columnconfigure(0, weight=1)

        self.box = ctk.CTkTextbox(c, wrap="word")
        self.box.grid(row=0, column=0, padx=14, pady=12, sticky="nsew")

        # Buttons
        row = ctk.CTkFrame(c, fg_color="transparent")
        row.grid(row=1, column=0, padx=14, pady=(0,12), sticky="ew")
        ctk.CTkButton(row, text="Export Log (TXT)", command=self._export).pack(side="left", padx=6)
        ctk.CTkButton(row, text="Clear", command=self._clear).pack(side="right", padx=6)

    # public sink for MainApp
    def append(self, msg: str):
        self.messages.append(msg)
        self.box.insert("end", msg + "\n")
        self.box.see("end")

    # actions
    def _export(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
            initialfile="pawdio-log.txt"
        )
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.messages))
        self.append(f"[EXPORT] log -> {path}")

    def _clear(self):
        self.messages.clear()
        self.box.delete("1.0", "end")
