import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")

import threading
import time
import math
import tkinter as tk
import queue
import traceback
import cv2
import mediapipe as mp
import numpy as np

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ---------------- Configuration ----------------
# Slouch detection thresholds
HEAD_FORWARD_Z_THRESHOLD = -0.07
TORSO_LEAN_THRESHOLD_DEG = 10.0
NECK_FLEX_THRESHOLD_DEG = 12.0

NOSE_DOWN_THRESHOLD = 0.04
COMPRESSION_THRESHOLD = 0.05
GROUP_DOWN_THRESHOLD = 0.025
GROUP_SYNC_CV_MAX = 0.15

CONSECUTIVE_BAD_FRAMES_TRIGGER = 15
CONSECUTIVE_GOOD_FRAMES_CLEAR = 10

SMOOTHING_WINDOW = 7
FRAME_INTERVAL = 0.03
UI_POLL_INTERVAL_MS = 33

# Baseline capture parameters
BASELINE_REQUIRED_STABLE_FRAMES = 50
BASELINE_MIN_FRAMES = 60
BASELINE_MOTION_MAX = 0.012
BASELINE_START_DELAY = 15

# Presence detection parameters
AWAY_FRAMES_THRESHOLD = 30
RETURN_STABLE_FRAMES = 20
PRESENCE_MIN_VISIBILITY = 0.5
CLEAR_ALERT_ON_AWAY = True
REQUIRE_RETURN_GRACE = True
RESET_BAD_COUNTS_ON_AWAY = True
OPTIONAL_FORCE_REBASELINE_ON_RETURN = False

# Sitting session timer parameters
SIT_SESSION_LIMIT_MIN = 45          # Minutes until MOVE alert
MICROBREAK_INTERVAL_MIN = 0         # Set >0 for microbreak nudges (e.g., 30); 0 disables
MIN_AWAY_BREAK_SECONDS = 120        # Away duration to count as a full break
REQUIRE_FULL_BREAK_TO_RESET = True  # If False, any away resets timer

# Sound / UI
DEBUG = False
ENABLE_SOUND = True

def dlog(*args):
    if DEBUG:
        print("DEBUG:", *args)

# ---------------- Utilities ----------------
def angle_with_vertical(v):
    vx, vy = v[0], v[1]
    dot = vx*0 + vy*(-1)
    mag_v = math.sqrt(vx*vx + vy*vy)
    if mag_v == 0:
        return 0.0
    return math.degrees(math.acos(max(min(dot / mag_v, 1), -1)))

def neck_angle(ear, shoulder):
    v = (shoulder.x - ear.x, shoulder.y - ear.y, shoulder.z - ear.z)
    return angle_with_vertical(v)

class RollingMetric:
    def __init__(self, window):
        self.window = window
        self.values = []
    def add(self, v):
        self.values.append(v)
        if len(self.values) > self.window:
            self.values.pop(0)
    @property
    def avg(self):
        return sum(self.values)/len(self.values) if self.values else 0.0

# ---------------- Sound Handling ----------------
class SoundPlayer:
    def __init__(self):
        self.stop_flag = threading.Event()
        self.thread = None
        # Get absolute path to beep.wav in same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.sound_file = os.path.join(script_dir, "beep.wav")
        self.mixer = None
    def start(self):
        if not ENABLE_SOUND:
            return
        if self.thread and self.thread.is_alive():
            return
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def stop(self):
        self.stop_flag.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
    def _loop(self):
        try:
            import pygame
            if self.mixer is None:
                pygame.mixer.init()
                self.mixer = pygame.mixer
        except ImportError:
            print("WARNING: pygame not installed. Install with: pip install pygame")
            return
        except Exception as e:
            print(f"pygame mixer init error: {e}")
            return
        
        if not os.path.exists(self.sound_file):
            print(f"WARNING: Sound file not found: {self.sound_file}")
            return
        
        try:
            sound = self.mixer.Sound(self.sound_file)
        except Exception as e:
            print(f"Sound loading error: {e}")
            return
            
        while not self.stop_flag.is_set():
            try:
                sound.play()
                # Wait for sound to finish or until stopped
                while sound.get_num_channels() > 0 and not self.stop_flag.is_set():
                    time.sleep(0.05)
            except Exception as e:
                print(f"Sound playback error: {e}")
                break
            # Pause between repeats
            for _ in range(5):
                if self.stop_flag.is_set():
                    break
                time.sleep(0.05)

# ---------------- Alert Manager ----------------
class AlertManager:
    def __init__(self, root, title, message):
        self.root = root
        self.popup = None
        self.sound = SoundPlayer()
        self.active = False
        self.title = title
        self.message = message
    def show(self):
        if self.active:
            return
        self.active = True
        try:
            self.popup = tk.Toplevel(self.root)
            self.popup.title(self.title)
            self.popup.configure(bg="black")
            self.popup.geometry("560x200+140+140")
            lbl = tk.Label(self.popup, text=self.message,
                           fg="red", bg="black", font=("Arial Black", 28), wraplength=520, justify="center")
            lbl.pack(expand=True, fill="both")
        except Exception as e:
            print("Popup creation error:", e)
        self.sound.start()
    def clear(self):
        if not self.active:
            return
        self.active = False
        try:
            if self.popup and self.popup.winfo_exists():
                self.popup.destroy()
        except Exception as e:
            print("Popup destroy error:", e)
        self.popup = None
        self.sound.stop()
    def shutdown(self):
        self.clear()

# ---------------- Worker ----------------
class PostureWorker:
    def __init__(self, frame_queue, status_queue, control):
        self.frame_queue = frame_queue
        self.status_queue = status_queue
        self.control = control
        self.stop_event = threading.Event()

        self.cap = None
        self.pose = None
        self.baseline = {}

        # Posture alerts
        self.posture_alert_active = False
        self.bad_frame_count = 0
        self.good_frame_count = 0

        # Rolling metrics
        self.roll_head_z = RollingMetric(SMOOTHING_WINDOW)
        self.roll_torso_angle = RollingMetric(SMOOTHING_WINDOW)
        self.roll_neck_angle = RollingMetric(SMOOTHING_WINDOW)
        self.roll_nose_y = RollingMetric(SMOOTHING_WINDOW)
        self.roll_shoulder_mid_y = RollingMetric(SMOOTHING_WINDOW)
        self.roll_hip_mid_y = RollingMetric(SMOOTHING_WINDOW)
        self.roll_compression = RollingMetric(SMOOTHING_WINDOW)

        # Baseline capture state
        self.baseline_capturing = False
        self.baseline_frames_total = 0
        self.baseline_stable_count = 0
        self.baseline_samples = []
        self.previous_landmarks = None
        self.start_frame_counter = 0

        # Presence state
        self.user_present = False
        self.away_frame_counter = 0
        self.present_frame_counter = 0
        self.return_grace_active = False
        self.return_grace_counter = 0

        # Sitting timer state
        self.sitting_accumulated_seconds = 0.0
        self.last_presence_timestamp = time.time()
        self.last_session_start_timestamp = time.time()
        self.last_microbreak_alert_time = None
        self.break_alert_active = False  # MOVE alert state
        self.microbreak_alert_active = False

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            return False
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.start_frame_counter = 0
        self.last_session_start_timestamp = time.time()
        self.last_presence_timestamp = time.time()
        return True

    def stop_camera(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.pose:
            try:
                self.pose.close()
            except Exception:
                pass
            self.pose = None

    def request_stop(self):
        self.stop_event.set()

    def request_baseline_capture(self):
        if not self.cap or not self.pose:
            self.status_queue.put("Camera not running.")
            return
        if self.baseline_capturing:
            return
        if self.start_frame_counter < BASELINE_START_DELAY:
            self.status_queue.put("Waiting for camera stabilization...")
            return
        if not self.user_present:
            self.status_queue.put("User not detected. Sit in frame to capture baseline.")
            return
        self.baseline_capturing = True
        self.baseline_frames_total = 0
        self.baseline_stable_count = 0
        self.baseline_samples = []
        self.previous_landmarks = None
        self.baseline = {}
        self.posture_alert_active = False
        self.bad_frame_count = 0
        self.good_frame_count = 0
        self.control["monitoring"] = False
        self.status_queue.put("Baseline capture started. Sit still...")

    def reset_sitting_timer(self):
        self.sitting_accumulated_seconds = 0.0
        self.last_session_start_timestamp = time.time()
        self.break_alert_active = False
        self.microbreak_alert_active = False
        self.status_queue.put("Sitting timer reset.")

    # Landmark & metrics
    def _extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None, res
        return res.pose_landmarks.landmark, res

    def _compute_posture_metrics(self, lm):
        try:
            NOSE = mp.solutions.pose.PoseLandmark.NOSE.value
            L_SH = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
            R_SH = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
            L_HP = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
            R_HP = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
            L_EAR = mp.solutions.pose.PoseLandmark.LEFT_EAR.value
            R_EAR = mp.solutions.pose.PoseLandmark.RIGHT_EAR.value

            nose = lm[NOSE]
            l_sh = lm[L_SH]; r_sh = lm[R_SH]
            l_hp = lm[L_HP]; r_hp = lm[R_HP]
            l_ear = lm[L_EAR] if lm[L_EAR].visibility > 0.5 else None
            r_ear = lm[R_EAR] if lm[R_EAR].visibility > 0.5 else None

            shoulder_mid_x = (l_sh.x + r_sh.x)/2.0
            shoulder_mid_y = (l_sh.y + r_sh.y)/2.0
            shoulder_mid_z = (l_sh.z + r_sh.z)/2.0

            hip_mid_x = (l_hp.x + r_hp.x)/2.0
            hip_mid_y = (l_hp.y + r_hp.y)/2.0
            hip_mid_z = (l_hp.z + r_hp.z)/2.0

            torso_vec = (shoulder_mid_x - hip_mid_x,
                         shoulder_mid_y - hip_mid_y,
                         shoulder_mid_z - hip_mid_z)
            torso_angle = angle_with_vertical(torso_vec)
            shoulder_hip_dist = abs(shoulder_mid_y - hip_mid_y)

            ear = None
            if l_ear and r_ear:
                ear = l_ear if l_ear.visibility >= r_ear.visibility else r_ear
            elif l_ear:
                ear = l_ear
            elif r_ear:
                ear = r_ear
            neck_ang = None
            if ear:
                shoulder_ref = l_sh if ear == l_ear else r_sh
                neck_ang = neck_angle(ear, shoulder_ref)

            metrics = {
                "nose_z": nose.z,
                "torso_angle": torso_angle,
                "nose_y": nose.y,
                "shoulder_mid_y": shoulder_mid_y,
                "hip_mid_y": hip_mid_y,
                "shoulder_hip_dist": shoulder_hip_dist
            }
            if neck_ang is not None:
                metrics["neck_angle"] = neck_ang
            return metrics
        except Exception as e:
            dlog("Compute posture metrics error:", e)
            return None

    def _frame_motion_score(self, lm):
        if self.previous_landmarks is None:
            self.previous_landmarks = lm
            return 0.0
        total = 0.0
        count = 0
        for prev, cur in zip(self.previous_landmarks, lm):
            total += abs(cur.x - prev.x) + abs(cur.y - prev.y)
            count += 1
        self.previous_landmarks = lm
        return total / max(count, 1)

    def _attempt_finalize_baseline(self):
        if self.baseline_stable_count >= BASELINE_REQUIRED_STABLE_FRAMES or \
           self.baseline_frames_total >= BASELINE_MIN_FRAMES:
            if not self.baseline_samples:
                self.status_queue.put("Baseline failed: no stable samples.")
            else:
                avg = {}
                keys = self.baseline_samples[0].keys()
                for k in keys:
                    avg[k] = sum(s[k] for s in self.baseline_samples) / len(self.baseline_samples)
                avg["baseline_shoulder_hip_dist"] = avg["shoulder_hip_dist"]
                self.baseline = avg
                self.roll_head_z.values = [avg["nose_z"]]
                self.roll_torso_angle.values = [avg["torso_angle"]]
                self.roll_nose_y.values = [avg["nose_y"]]
                self.roll_shoulder_mid_y.values = [avg["shoulder_mid_y"]]
                self.roll_hip_mid_y.values = [avg["hip_mid_y"]]
                self.roll_compression.values = [0.0]
                if "neck_angle" in avg:
                    self.roll_neck_angle.values = [avg["neck_angle"]]
                self.status_queue.put(
                    f"Baseline complete. head_z={avg['nose_z']:.3f}, torso={avg['torso_angle']:.1f}°, nose_y={avg['nose_y']:.3f}"
                )
            self.baseline_capturing = False

    def _evaluate_posture(self, metrics):
        if not self.baseline or not self.control["monitoring"] or not self.user_present or self.return_grace_active:
            return False, [], {}
        flags = []
        self.roll_head_z.add(metrics["nose_z"])
        self.roll_torso_angle.add(metrics["torso_angle"])
        self.roll_nose_y.add(metrics["nose_y"])
        self.roll_shoulder_mid_y.add(metrics["shoulder_mid_y"])
        self.roll_hip_mid_y.add(metrics["hip_mid_y"])

        compression = 0.0
        if "baseline_shoulder_hip_dist" in self.baseline and self.baseline["baseline_shoulder_hip_dist"] > 1e-6:
            compression = (self.baseline["baseline_shoulder_hip_dist"] - metrics["shoulder_hip_dist"]) / self.baseline["baseline_shoulder_hip_dist"]
        self.roll_compression.add(compression)

        if "neck_angle" in metrics:
            self.roll_neck_angle.add(metrics["neck_angle"])

        sm_head_z = self.roll_head_z.avg
        sm_torso_angle = self.roll_torso_angle.avg
        sm_nose_y = self.roll_nose_y.avg
        sm_sh_y = self.roll_shoulder_mid_y.avg
        sm_hp_y = self.roll_hip_mid_y.avg
        sm_compression = self.roll_compression.avg
        sm_neck_angle = self.roll_neck_angle.avg if "neck_angle" in metrics else None

        head_z_delta = sm_head_z - self.baseline.get("nose_z", sm_head_z)
        if head_z_delta < HEAD_FORWARD_Z_THRESHOLD:
            flags.append("head_forward")

        torso_delta = sm_torso_angle - self.baseline.get("torso_angle", sm_torso_angle)
        if torso_delta > TORSO_LEAN_THRESHOLD_DEG:
            flags.append("torso_lean")

        neck_delta = 0.0
        if "neck_angle" in self.baseline and sm_neck_angle is not None:
            neck_delta = sm_neck_angle - self.baseline["neck_angle"]
            if neck_delta > NECK_FLEX_THRESHOLD_DEG:
                flags.append("neck_flex")

        nose_y_delta = sm_nose_y - self.baseline.get("nose_y", sm_nose_y)
        sh_y_delta = sm_sh_y - self.baseline.get("shoulder_mid_y", sm_sh_y)
        hp_y_delta = sm_hp_y - self.baseline.get("hip_mid_y", sm_hp_y)
        group_deltas = [nose_y_delta, sh_y_delta, hp_y_delta]
        positive_group = all(d > GROUP_DOWN_THRESHOLD for d in group_deltas)
        cv = 0.0
        if positive_group:
            mean = sum(group_deltas)/3.0
            var = sum((d-mean)**2 for d in group_deltas)/3.0
            std = math.sqrt(var)
            cv = std / mean if mean > 1e-6 else 0.0

        if sm_compression > COMPRESSION_THRESHOLD and nose_y_delta > NOSE_DOWN_THRESHOLD:
            flags.append("compression_slouch")
        if positive_group and cv < GROUP_SYNC_CV_MAX and nose_y_delta > NOSE_DOWN_THRESHOLD:
            flags.append("vertical_slouch")

        delta_dict = {
            "head_z_delta": head_z_delta,
            "torso_delta": torso_delta,
            "neck_delta": neck_delta,
            "nose_y_delta": nose_y_delta,
            "compression": sm_compression
        }
        return len(flags) > 0, flags, delta_dict

    # Presence
    def _update_presence(self, lm):
        now = time.time()
        if lm is None:
            self.away_frame_counter += 1
            self.present_frame_counter = 0
        else:
            NOSE = mp.solutions.pose.PoseLandmark.NOSE.value
            nose_vis = lm[NOSE].visibility
            if nose_vis >= PRESENCE_MIN_VISIBILITY:
                self.present_frame_counter += 1
                self.away_frame_counter = 0
            else:
                self.away_frame_counter += 1
                self.present_frame_counter = 0

        # Transition to away
        if self.user_present and self.away_frame_counter >= AWAY_FRAMES_THRESHOLD:
            self.user_present = False
            self.return_grace_active = False
            if CLEAR_ALERT_ON_AWAY:
                self.posture_alert_active = False
                self.break_alert_active = False
                self.microbreak_alert_active = False
            if RESET_BAD_COUNTS_ON_AWAY:
                self.bad_frame_count = 0
                self.good_frame_count = 0
            self.last_presence_timestamp = now
            self.status_queue.put("User away")

        # Transition to present
        if not self.user_present and self.present_frame_counter >= RETURN_STABLE_FRAMES:
            self.user_present = True
            self.last_presence_timestamp = now
            if REQUIRE_RETURN_GRACE:
                self.return_grace_active = True
                self.return_grace_counter = RETURN_STABLE_FRAMES
                self.status_queue.put("User returned (stabilizing)")
            else:
                self.return_grace_active = False
                self.status_queue.put("User present")
            # Reset timers if away long enough
            away_duration = now - self.last_session_start_timestamp if not self.user_present else 0
            if REQUIRE_FULL_BREAK_TO_RESET:
                if (now - self.last_presence_timestamp) >= MIN_AWAY_BREAK_SECONDS:
                    self.reset_sitting_timer()
            else:
                self.reset_sitting_timer()

    def _process_return_grace(self):
        if self.return_grace_active:
            self.return_grace_counter -= 1
            if self.return_grace_counter <= 0:
                self.return_grace_active = False
                self.status_queue.put("Presence stabilized; monitoring resumes.")

    # Sitting timer updates
    def _update_sitting_timer(self):
        # Only count when present, baseline done, not capturing baseline, not in grace
        if (self.user_present and self.baseline and not self.baseline_capturing
                and not self.return_grace_active):
            self.sitting_accumulated_seconds += FRAME_INTERVAL
            # Break alert
            if not self.break_alert_active and self.sitting_accumulated_seconds >= SIT_SESSION_LIMIT_MIN * 60:
                self.break_alert_active = True
                self.status_queue.put("Move alert triggered (session limit reached).")
            # Microbreak alert
            if MICROBREAK_INTERVAL_MIN > 0:
                if (self.last_microbreak_alert_time is None or
                        (time.time() - self.last_microbreak_alert_time) >= MICROBREAK_INTERVAL_MIN * 60):
                    # Trigger microbreak (but skip if within last full break alert)
                    self.microbreak_alert_active = True
                    self.last_microbreak_alert_time = time.time()
                    self.status_queue.put("Microbreak reminder.")
        else:
            # If user away and away long enough -> reset timer (handled on presence transition)
            pass

    def run(self):
        try:
            if not self.start_camera():
                self.status_queue.put("Cannot open webcam.")
                return
            self.status_queue.put("Camera started. Press 'Capture Baseline' after a moment.")
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    self.status_queue.put("Frame read failure.")
                    break
                self.start_frame_counter += 1

                lm, res = self._extract_landmarks(frame)
                self._update_presence(lm)
                if self.user_present and self.return_grace_active:
                    self._process_return_grace()

                metrics = None
                if lm:
                    metrics = self._compute_posture_metrics(lm)

                # Baseline capture
                if self.baseline_capturing and metrics and self.user_present and not self.return_grace_active:
                    motion = self._frame_motion_score(lm)
                    self.baseline_frames_total += 1
                    stable = motion < BASELINE_MOTION_MAX
                    if stable:
                        self.baseline_samples.append(metrics)
                        self.baseline_stable_count += 1
                    self.status_queue.put(
                        f"Baseline: {self.baseline_stable_count}/{BASELINE_REQUIRED_STABLE_FRAMES} stable "
                        f"(motion={motion:.4f})"
                    )
                    if (self.baseline_stable_count >= BASELINE_REQUIRED_STABLE_FRAMES or
                        self.baseline_frames_total >= BASELINE_MIN_FRAMES):
                        self._attempt_finalize_baseline()
                elif self.baseline_capturing and not self.user_present:
                    self.status_queue.put("Baseline paused: user away.")

                # Evaluate posture
                flags = []
                delta_dict = {}
                slouching = False
                if metrics and self.baseline and not self.baseline_capturing:
                    slouching, flags, delta_dict = self._evaluate_posture(metrics)
                    if (self.user_present and not self.return_grace_active
                            and self.control["monitoring"]):
                        if slouching:
                            self.bad_frame_count += 1
                            self.good_frame_count = 0
                        else:
                            self.good_frame_count += 1
                            self.bad_frame_count = 0

                        if (not self.posture_alert_active and
                            self.bad_frame_count >= CONSECUTIVE_BAD_FRAMES_TRIGGER):
                            self.posture_alert_active = True
                        if (self.posture_alert_active and
                            self.good_frame_count >= CONSECUTIVE_GOOD_FRAMES_CLEAR):
                            self.posture_alert_active = False
                    else:
                        if CLEAR_ALERT_ON_AWAY and not self.user_present:
                            self.posture_alert_active = False

                # Sitting timer update
                self._update_sitting_timer()
                if not self.user_present:
                    # Clear break/microbreak alerts while away
                    self.break_alert_active = False
                    self.microbreak_alert_active = False

                # Build status
                parts = []
                if not self.user_present:
                    parts.append("UserAway")
                elif self.return_grace_active:
                    parts.append("ReturnGrace")
                # Timer display
                parts.append(f"Sitting:{int(self.sitting_accumulated_seconds//60)}m {int(self.sitting_accumulated_seconds%60)}s / Limit:{SIT_SESSION_LIMIT_MIN}m")
                if self.baseline and delta_dict and not self.baseline_capturing and self.user_present and not self.return_grace_active:
                    parts.append(f"ΔHeadZ:{delta_dict['head_z_delta']:+.3f}")
                    parts.append(f"ΔTorso:{delta_dict['torso_delta']:+.1f}°")
                    parts.append(f"ΔNoseY:{delta_dict['nose_y_delta']:+.3f}")
                    parts.append(f"Comp:{delta_dict['compression']*100:5.1f}%")
                    if "neck_angle" in self.baseline:
                        parts.append(f"ΔNeck:{delta_dict['neck_delta']:+.1f}°")
                elif not self.baseline_capturing and self.user_present:
                    parts.append("Baseline not set.")
                if flags and self.control["monitoring"] and self.user_present and not self.return_grace_active:
                    parts.append("Flags:" + ",".join(flags))
                parts.append(f"Monitoring:{'ON' if self.control['monitoring'] else 'OFF'}")
                if self.posture_alert_active:
                    parts.append("PostureALERT")
                if self.break_alert_active:
                    parts.append("MoveALERT")
                if self.microbreak_alert_active:
                    parts.append("MicroBreak")
                if self.baseline_capturing:
                    parts.append("CapturingBaseline")
                status_text = " | ".join(parts)
                if not self.status_queue.full():
                    self.status_queue.put(status_text)

                # Draw landmarks
                if lm and self.control["draw_landmarks"]:
                    try:
                        h, w = frame.shape[:2]
                        idxs = [
                            mp.solutions.pose.PoseLandmark.NOSE.value,
                            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
                            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
                            mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
                            mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
                        ]
                        color = (0, 255, 0) if self.user_present else (0, 0, 255)
                        if self.return_grace_active:
                            color = (0, 165, 255)
                        for i in idxs:
                            p = lm[i]
                            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5,
                                       color if not self.baseline_capturing else (0, 128, 255), -1)
                    except Exception as e:
                        dlog("Landmark draw error:", e)

                # Decide which alert to show (priority: move alert > posture alert > microbreak)
                effective_alert = None
                if self.break_alert_active:
                    effective_alert = "MOVE"
                elif self.posture_alert_active:
                    effective_alert = "POSTURE"
                elif self.microbreak_alert_active:
                    effective_alert = "MICRO"

                if not self.frame_queue.full():
                    show_alert = (effective_alert is not None and
                                  self.user_present and not self.return_grace_active)
                    self.frame_queue.put((frame, effective_alert if show_alert else None))

                time.sleep(FRAME_INTERVAL)
        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.status_queue.put(f"Worker crashed: {e}")
            print("Worker thread exception:\n", err)
        finally:
            self.stop_camera()
            dlog("Worker exiting.")

# ---------------- App ----------------
class PostureApp:
    def __init__(self, root):
        self.root = root
        root.title("Posture Monitor (Presence + Sitting Timer)")

        self.control = {
            "monitoring": False,
            "draw_landmarks": True
        }

        self.frame_queue = queue.Queue(maxsize=2)
        self.status_queue = queue.Queue(maxsize=60)

        self.worker = None
        self.worker_thread = None

        # Separate alert managers (reuse same class)
        self.posture_alert = AlertManager(root, "Posture Warning", "ADJUST POSTURE PLEASE!")
        self.move_alert = AlertManager(root, "Movement Reminder", "TIME TO MOVE! STAND UP AND STRETCH")
        self.microbreak_alert = AlertManager(root, "Microbreak Reminder", "TAKE A SHORT MICROBREAK: MOVE OR STRETCH")

        self.status_var = tk.StringVar(value="Idle.")
        self.current_photo = None
        self.closed = False

        self._build_ui()
        self._poll_queues()

    def _build_ui(self):
        top = tk.Frame(self.root, padx=10, pady=10)
        top.pack(fill="both", expand=True)

        self.btn_start = tk.Button(top, text="Start Camera", width=15, command=self.start_camera)
        self.btn_start.grid(row=0, column=0, padx=5, pady=5)

        self.btn_baseline = tk.Button(top, text="Capture Baseline", width=15,
                                      command=self.capture_baseline, state="disabled")
        self.btn_baseline.grid(row=0, column=1, padx=5, pady=5)

        self.btn_monitor = tk.Button(top, text="Start Monitoring", width=15,
                                     command=self.toggle_monitoring, state="disabled")
        self.btn_monitor.grid(row=0, column=2, padx=5, pady=5)

        self.btn_reset_timer = tk.Button(top, text="Reset Sitting Timer", width=15,
                                         command=self.reset_sitting_timer, state="disabled")
        self.btn_reset_timer.grid(row=0, column=3, padx=5, pady=5)

        self.btn_stop = tk.Button(top, text="Stop Camera", width=15, command=self.stop_camera, state="disabled")
        self.btn_stop.grid(row=0, column=4, padx=5, pady=5)

        self.chk_landmarks = tk.Checkbutton(top, text="Draw Landmarks",
                                            command=self.toggle_landmarks, indicatoron=True)
        self.chk_landmarks.grid(row=0, column=5, padx=5, pady=5)

        status_lbl = tk.Label(top, textvariable=self.status_var, anchor="w",
                              width=140, relief="sunken")
        status_lbl.grid(row=1, column=0, columnspan=6, sticky="we", pady=8)

        self.video_label = tk.Label(top, text="(Video will appear here)")
        self.video_label.grid(row=2, column=0, columnspan=6, pady=10)

        info = tk.Label(top,
                        text="Workflow:\n1. Start Camera (wait stabilization)\n2. Capture Baseline\n3. Start Monitoring\nSession alert after sitting limit; leaving desk pauses & can reset.\nAdjust thresholds in code.",
                        justify="left")
        info.grid(row=3, column=0, columnspan=6, sticky="w")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_camera(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.set_status("Camera already running.")
            return
        self.worker = PostureWorker(self.frame_queue, self.status_queue, self.control)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()
        self.set_status("Starting camera...")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_baseline.configure(state="normal")
        self.btn_reset_timer.configure(state="normal")
        self.btn_monitor.configure(state="disabled")

    def stop_camera(self):
        if not self.worker:
            return
        self.control["monitoring"] = False
        self.worker.request_stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        self.worker_thread = None
        self.worker = None
        self.posture_alert.clear()
        self.move_alert.clear()
        self.microbreak_alert.clear()
        self.set_status("Camera stopped.")
        self.btn_start.configure(state="normal")
        self.btn_baseline.configure(state="disabled")
        self.btn_monitor.configure(state="disabled")
        self.btn_reset_timer.configure(state="disabled")
        self.btn_stop.configure(state="disabled")

    def capture_baseline(self):
        if not self.worker or not self.worker.cap:
            self.set_status("Start camera first.")
            return
        self.btn_baseline.configure(state="disabled")
        self.btn_monitor.configure(state="disabled")
        self.worker.request_baseline_capture()

    def toggle_monitoring(self):
        if not self.worker or not self.worker.baseline:
            self.set_status("Baseline not set.")
            return
        self.control["monitoring"] = not self.control["monitoring"]
        self.btn_monitor.configure(
            text="Stop Monitoring" if self.control["monitoring"] else "Start Monitoring"
        )
        self.set_status(f"Monitoring {'ON' if self.control['monitoring'] else 'OFF'}")

    def reset_sitting_timer(self):
        if not self.worker:
            return
        self.worker.reset_sitting_timer()

    def toggle_landmarks(self):
        self.control["draw_landmarks"] = not self.control["draw_landmarks"]
        self.set_status(f"Draw landmarks {'ON' if self.control['draw_landmarks'] else 'OFF'}")

    def set_status(self, msg):
        self.status_var.set(msg)

    def _poll_queues(self):
        try:
            while not self.status_queue.empty():
                msg = self.status_queue.get_nowait()
                self.set_status(msg)
                if "Baseline complete" in msg:
                    self.btn_monitor.configure(state="normal")
                if "Baseline failed" in msg:
                    self.btn_baseline.configure(state="normal")
                if "Sitting timer reset" in msg:
                    pass
        except Exception as e:
            dlog("Status queue error:", e)

        try:
            if not self.frame_queue.empty():
                frame, alert_type = self.frame_queue.get_nowait()
                self._update_video(frame)
                # Manage alerts based on alert_type
                if alert_type == "MOVE":
                    if not self.move_alert.active:
                        self.move_alert.show()
                    # Clear others
                    self.posture_alert.clear()
                    self.microbreak_alert.clear()
                elif alert_type == "POSTURE":
                    if not self.posture_alert.active:
                        self.posture_alert.show()
                    self.move_alert.clear()
                    self.microbreak_alert.clear()
                elif alert_type == "MICRO":
                    if not self.microbreak_alert.active:
                        self.microbreak_alert.show()
                    # Do not clear posture or move if they are active (prioritize higher)
                    if self.move_alert.active:
                        self.microbreak_alert.clear()
                else:
                    # No alert
                    self.posture_alert.clear()
                    self.move_alert.clear()
                    self.microbreak_alert.clear()
        except Exception as e:
            print("Frame queue processing error:", e)

        if not self.closed:
            self.root.after(UI_POLL_INTERVAL_MS, self._poll_queues)

    def _update_video(self, frame):
        if not PIL_AVAILABLE:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self.current_photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=self.current_photo)
        except Exception as e:
            dlog("Video update error:", e)

    def on_close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.stop_camera()
        except Exception as e:
            print("Error during stop_camera on close:", e)
        try:
            self.posture_alert.shutdown()
            self.move_alert.shutdown()
            self.microbreak_alert.shutdown()
        except Exception as e:
            print("AlertManager shutdown error:", e)
        self.root.destroy()

def main():
    root = tk.Tk()
    PostureApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()