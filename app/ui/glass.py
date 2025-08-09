# app/ui/glass.py
import os, sys
import customtkinter as ctk
from PIL import Image, ImageFilter, ImageEnhance, ImageTk  # <- ImageTk is important
from tkinter import PhotoImage

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

def make_background(path, size, blur=18, darken=0.18, desaturate=0.05):
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
            # Use ImageTk.PhotoImage, not tk.PhotoImage
            self._photo = ImageTk.PhotoImage(img)
            self.configure(image=self._photo)
        except Exception:
            pass

class GlassCard(ctk.CTkFrame):
    """
    'Glass' card:
    - parent frame is transparent so background shows through
    - inner frame has a faint frosted color + subtle border
    """
    def __init__(self, master, **kwargs):
        # IMPORTANT: pass a SINGLE 'transparent' string
        super().__init__(master, corner_radius=18, fg_color="transparent", **kwargs)

        # inner frosted layer (no real alpha in CTk; choose soft light/dark colors)
        self.inner = ctk.CTkFrame(
            self,
            corner_radius=18,
            fg_color=("#F3F6F91A", "#0E11141A"),  # light/dark soft tint; CTk accepts short alpha in hex on recent versions; if not, swap to "#F3F6F9" / "#0E1114"
            border_width=1,
            border_color=("#FFFFFF33", "#FFFFFF22"),
        )
        self.inner.pack(fill="both", expand=True, padx=1, pady=1)

        # subtle top shine
        self.shine = ctk.CTkFrame(self.inner, height=1, fg_color=("#FFFFFF55", "#FFFFFF22"))
        self.shine.pack(fill="x", side="top")
