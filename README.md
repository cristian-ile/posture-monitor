# Posture Monitor (Improved)

## Run
```bash
# On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python posture_monitor.py

# On Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python posture_monitor.py
```

1. Click "Start Camera".
2. Sit upright; click "Capture Baseline".
3. After baseline capture succeeds you will see metrics and the "Start Monitoring" button enables (or toggles to ON).
4. Slouch (lean head forward) to trigger red popup + sound.
5. Correct posture to dismiss.

## New Detection
- Uses head depth (MediaPipe Z) rather than horizontal X for forward head movement.
- Uses torso lean angle (hip→shoulder vector vs vertical).
- Optional neck flex if ear visible.
- Rolling average smoothing.

## Tuning
Edit in `posture_monitor.py`:
```python
HEAD_FORWARD_Z_THRESHOLD = -0.07
TORSO_LEAN_THRESHOLD_DEG = 10.0
NECK_FLEX_THRESHOLD_DEG = 12.0
CONSECUTIVE_BAD_FRAMES_TRIGGER = 15
CONSECUTIVE_GOOD_FRAMES_CLEAR = 10
SMOOTHING_WINDOW = 7
```

Make threshold magnitude smaller for stricter (e.g. -0.05), larger for lenient (e.g. -0.10).

## Troubleshooting
- Ensure face and upper torso visible.
- Good lighting reduces landmark jitter.
- If baseline metrics show zeros, retry baseline.
- Use console prints for extra debug.

## Privacy
All processing local. No frames saved.
