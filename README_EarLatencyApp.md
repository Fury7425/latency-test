
# Ear/Headset Latency Tester (GUI)

Builds on your existing approach (test signals + cross-correlation with sub-sample interpolation) but adds:
- **Per-sound calibration** (each preset stores its own baseline ms offset)
- **Global system offset** (impulse-based; best with direct loopback cable)
- **Run tests individually or all at once**
- **Device selection** (input/output)
- **Save plots & text reports**
- **JSON config** persisted automatically

## Quickstart

1) Install Python 3.10+ (Windows/macOS/Linux).  
2) Install deps:
```bash
pip install -r requirements.txt
```
> Windows tip if PyAudio fails:  
> `pip install pipwin && pipwin install pyaudio`

3) Run the app:
```bash
python ear_latency_app.py
```

## Usage

- **Settings / Devices**: choose your input (mic) and output (headset) devices and set sample rate/duration. Click **Apply** and **Use Selected Devices**.
- **Calibration**:
  - **Calibrate GLOBAL System Offset (Impulse)**: for measuring soundcard/driver latency; **use a loopback cable** if you can (headphone out → mic in).
  - **Calibrate Selected/All Presets**: saves a **per-sound baseline** for each preset (1k/5k/200Hz/2k/Impulse).
- **Tests**: select presets and click **Run Selected** or **Run ALL**. Choose repeats, output folder, and whether to save plots and an overall bar chart.
- **Report**: after a run, click **Save Text Report** to write a detailed summary.

The app subtracts:  
`calibrated_delay = raw_delay - (global_system_offset_ms + per_sound_baseline_ms)`

## Notes

- Place the mic very close to the earcup (or inside an earbud seal) to minimize air delay and room reflections.
- Wireless products add codec/stack buffering; expect tens of ms or more.
- If calibrated delay is negative, redo calibration (or reduce the per-sound baseline).

