
import customtkinter as ctk

def apply_theme(cfg_ui):
    ctk.set_appearance_mode(cfg_ui.get("appearance_mode", "Dark"))
    ctk.set_default_color_theme(cfg_ui.get("color_theme", "blue"))
