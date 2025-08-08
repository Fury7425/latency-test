
import os, json

CALIBRATION_JSON = "calibration_per_sound.json"

def default_config():
    return {
        "global_system_offset_ms": 0.0,
        "per_sound_offsets_ms": {},
        "last_settings": {
            "sample_rate": 44100,
            "duration": 0.5,
            "repeats": 5,
            "input_device_index": None,
            "output_device_index": None,
            "output_dir": os.getcwd(),
        },
        "ui": {
            "appearance_mode": "Dark",
            "color_theme": "blue",
            "labs_enabled": True
        }
    }

def load_config(path=CALIBRATION_JSON):
    cfg = default_config()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception:
            pass
    cfg.setdefault("per_sound_offsets_ms", {})
    cfg.setdefault("ui", {}).setdefault("labs_enabled", True)
    return cfg

def save_config(cfg, path=CALIBRATION_JSON):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False
