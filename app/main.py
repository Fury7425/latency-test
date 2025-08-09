# main.py
"""
Entry point for Pawdio‑Lab.
Works whether the project is run as a package or flattened.
"""

import sys

def _run():
    # Prefer package import (app/...), fall back to flat layout if needed.
    try:
        from app.ui.app import MainApp  # package layout
    except Exception:
        # If someone flattened the project, try relative path variant
        from ui.app import MainApp  # flat layout

    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    _run()
