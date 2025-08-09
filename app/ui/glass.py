# app/ui/glass.py
import os, sys
import customtkinter as ctk
from PIL import Image, ImageFilter, ImageEnhance, ImageTk  # Pillow is required

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

def make_background(path, size, blur=18, darken=0.18, desaturate=0.05):
    """Return a blurred/dimmed PIL.Image matching the given size."""
    img = Image.open(path).convert("RGB")
    w, h = size
    img = img.resize((max(1, w), max(1, h)), Image.LANCZOS)
    if desaturate > 0:
        gray = img.convert("L").convert("RGB")
        img = Image.blend(img, gray, desaturate)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if darken > 0:
        img = ImageEnhance.Brightness(img).enhance(1.0 - darken)
    return img

class Background(ctk.CTkLabel):
    """Full‑window blurred background that resizes with the window."""
    def __init__(self, master, image_path: str, **kwargs):
        super().__init__(master, text="", **kwargs)
        self._image_path = image_path
        self._photo = None   # keep a ref to avoid GC
        self.bind("<Configure>", self._on_resize)
        self._on_resize()

    def _on_resize(self, *_):
        w = max(300, self.winfo_width() or self.winfo_reqwidth())
        h = max(300, self.winfo_height() or self.winfo_reqheight())
        try:
            img = make_background(self._image_path, (w, h))
            self._photo = ImageTk.PhotoImage(img)
            self.configure(image=self._photo)
        except Exception:
            pass

class GlassCard(ctk.CTkFrame):
    """
    'Glass' card using only solid colors (no alpha hex):
    - outer frame is transparent so the blurred bg shows through
    - inner frame uses very light/dark solids to *suggest* translucency
    """
    def __init__(self, master, **kwargs):
        # IMPORTANT: single 'transparent' string (not a tuple)
        super().__init__(master, corner_radius=18, fg_color="transparent", **kwargs)

        # Choose gentle solids for light/dark modes (no alpha)
        inner_fg = ("#F5F7FA", "#171A1F")      # light / dark
        border_fg = ("#E2E8F0", "#2A2F3A")     # subtle border
        shine_fg  = ("#FFFFFF", "#2B2F36")     # thin top "shine" divider

        self.inner = ctk.CTkFrame(
            self,
            corner_radius=18,
            fg_color=inner_fg,
            border_width=1,
            border_color=border_fg,
        )
        self.inner.pack(fill="both", expand=True, padx=1, pady=1)

        # Subtle top shine line (solid color only)
        self.shine = ctk.CTkFrame(self.inner, height=1, fg_color=shine_fg)
        self.shine.pack(fill="x", side="top")
