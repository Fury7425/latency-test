# app/ui/glass.py
import os, sys
import customtkinter as ctk
from PIL import Image, ImageFilter, ImageEnhance
from tkinter import PhotoImage

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

def make_background(path, size, blur=18, darken=0.18, desaturate=0.05):
    img = Image.open(path).convert("RGB")
    w, h = size
    img = img.resize((max(1,w), max(1,h)), Image.LANCZOS)
    if desaturate > 0:
        # quick desaturation via grayscale mix
        gray = img.convert("L").convert("RGB")
        img = Image.blend(img, gray, desaturate)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if darken > 0:
        img = ImageEnhance.Brightness(img).enhance(1.0 - darken)
    return img

class Background(ctk.CTkLabel):
    """A full-window background label holding a blurred image (cached as PhotoImage)."""
    def __init__(self, master, image_path: str, **kwargs):
        super().__init__(master, text="", **kwargs)
        self._image_path = image_path
        self._photo = None
        self.bind("<Configure>", self._on_resize)
        self._on_resize()

    def _on_resize(self, *_):
        w = max(300, self.winfo_width() or self.winfo_reqwidth())
        h = max(300, self.winfo_height() or self.winfo_reqheight())
        try:
            img = make_background(self._image_path, (w, h))
            self._photo = PhotoImage(img)  # CTk accepts tk.PhotoImage
            self.configure(image=self._photo)
        except Exception:
            pass

class GlassCard(ctk.CTkFrame):
    """
    A simple "glassmorphism" card:
    - transparent bg so the blurred background shows through
    - subtle border + inner shine
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, corner_radius=18, fg_color=("transparent","transparent"), **kwargs)
        # inner layer to simulate frost
        self.inner = ctk.CTkFrame(
            self,
            corner_radius=18,
            fg_color=("#ffffff15", "#00000015"),  # light/dark semi-transparent overlay
            border_width=1,
            border_color=("#ffffff33", "#ffffff22"),
        )
        self.inner.pack(fill="both", expand=True, padx=1, pady=1)
        # optional top "shine" line
        self.shine = ctk.CTkFrame(self.inner, height=1, fg_color=("#ffffff55", "#ffffff22"))
        self.shine.pack(fill="x", side="top")
