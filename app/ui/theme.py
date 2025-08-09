# app/ui/theme.py
import customtkinter as ctk

def apply_theme(cfg_ui):
    # Appearance
    ctk.set_appearance_mode(cfg_ui.get("appearance_mode", "Dark"))
    # Accent color (keep “blue” — works well on glass)
    ctk.set_default_color_theme(cfg_ui.get("color_theme", "blue"))

    # Global typography (optional small lift)
    ctk.FontManager.load_font("Roboto")
