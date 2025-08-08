
import os, numpy as np

def safe_mean(a):
    vals = [x for x in a if x is not None]
    return float(np.mean(vals)) if vals else None

def safe_std(a):
    vals = [x for x in a if x is not None]
    return float(np.std(vals)) if vals else None

def rms(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x**2)) if x.size > 0 else 0.0

def dbfs(x):
    r = rms(x)
    return 20*np.log10(max(r, 1e-12))

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d
