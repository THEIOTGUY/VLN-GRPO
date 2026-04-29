#!/usr/bin/env python3
"""
ActiveVLN → Unitree Go1 real-robot deployment.

Connects to a running vLLM checkpoint server (same OpenAI-compatible API used
during eval) and drives a Unitree Go1 via the unitree_legged_sdk HighCmd UDP
interface.

New capabilities vs. the baseline:
  - Closed-loop obstacle replanning: depth sensor checked after every physical
    action; when an obstacle is detected the next VLM call uses a replanning
    prompt prefix instead of the normal "After that, the observation is:" prefix.
  - Spatial memory module: lightweight dead-reckoning map built from forward /
    turn odometry; a compact text summary is appended to every VLM prompt.
  - Semantic landmark tracking: landmark phrases are extracted from the
    instruction at startup; expected waypoints are logged as the agent
    progresses and included in the memory summary.

Usage
-----
# 1. Start the checkpoint server (from repo root):
#    vllm serve checkpoints/Qwen2.5-VL-3B_rl_r2r_4000 \\
#        --task generate --trust-remote-code \\
#        --limit-mm-per-prompt image=200,video=0 \\
#        --mm-processor-kwargs '{"max_pixels": 80000}' \\
#        --max-model-len 32768 --enable-prefix-caching \\
#        --disable-log-requests --port 8003

# 2. Run navigation (live camera + real robot):
#    python deploy/go1_nav.py --instruction "Walk to the door and stop"

# 3. With obstacle replanning + memory (requires RealSense depth):
#    python deploy/go1_nav.py --instruction "..." --rs \\
#        --enable-obstacle-replanning --enable-memory

# 4. Dry-run with a saved image (no robot / no camera):
#    python deploy/go1_nav.py --instruction "..." --image-path obs.jpg --dry-run
"""

import argparse
import base64
import glob as _glob
import json
import math
import os
import re
import sys
import time
import threading
import subprocess
import signal
import urllib.request
import hashlib
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from qwen_vl_utils.vision_process import smart_resize as _qwen_smart_resize
except ImportError:
    _qwen_smart_resize = None

# ── unitree SDK path (arm64 Jetson / AGX Orin) ──────────────────────────────
_SDK_LIB = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "unitree_legged_sdk", "lib", "python", "arm64")

# ── observation history (shared with web_ui.py via /tmp) ─────────────────────
_OBS_DIR   = "/tmp/vln_obs"
_HIST_FILE = "/tmp/vln_history.json"
_E_STOP_FLAG = "/tmp/vln_estop.flag"


def _clear_obs_history():
    for f in _glob.glob(f"{_OBS_DIR}/*.jpg"):
        try: os.remove(f)
        except OSError: pass
    try: os.remove(_HIST_FILE)
    except OSError: pass


def _append_obs(n: int, bgr: np.ndarray, actions: List[Tuple[int, float]], raw: str):
    if cv2 is None:
        return
    os.makedirs(_OBS_DIR, exist_ok=True)
    cv2.imwrite(f"{_OBS_DIR}/{n:04d}.jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    labels = []
    for aid, val in actions:
        if aid == 0:   labels.append("STOP")
        elif aid == 1: labels.append(f"FWD {val:.0f}cm")
        elif aid == 2: labels.append(f"LEFT {val:.0f}°")
        elif aid == 3: labels.append(f"RIGHT {val:.0f}°")
    try:
        hist = json.loads(open(_HIST_FILE).read()) if os.path.exists(_HIST_FILE) else []
    except Exception:
        hist = []
    hist.append({"n": n, "actions": labels, "raw": raw})
    with open(_HIST_FILE, "w") as fh:
        json.dump(hist, fh)

# ── prompts (verbatim from eval/vlnce/eval_vlnce.py) ────────────────────────
SYSTEM_PROMPT_R2R = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), "
    "or post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, "
    "move forward 75cm, turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, "
    "turn right 15 degrees, turn right 30 degrees, turn right 45 degrees, or stop. \n"
    "The instruction will be provided with each observation. "
    "You can take multiple actions at each turn. "
)

SYSTEM_PROMPT_RXR = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), "
    "or post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, "
    "move forward 75cm, turn left 30 degrees, turn left 60 degrees, turn left 90 degrees, "
    "turn right 30 degrees, turn right 60 degrees, turn right 90 degrees, or stop. \n"
    "The instruction will be provided with each observation. "
    "You can take multiple actions at each turn. "
)

USER_PROMPT = (
    "Instruction: {instruction}\n"
    "{memory}"
    "Decide your next action. "
    "You can take up to 3 actions at a time, separated by ','. "
)

# Shared /tmp flag files written by go1_nav.py and read by web_ui.py
_MEMORY_FILE  = "/tmp/vln_memory.json"    # current spatial memory state
_OBSTACLE_FILE = "/tmp/vln_obstacle.flag" # present when obstacle replanning fired


# ── Lightweight dead-reckoning spatial memory (deploy-only) ─────────────────
# Mirrors vlnce_server/memory.py logic but without the Ray/Habitat dependency.

_LANDMARK_COLOURS = (
    "red|blue|green|brown|white|black|gray|grey|yellow|orange|pink|purple"
    "|wooden|dark|light|beige|tan|teal|maroon|navy"
)
_LANDMARK_OBJECTS = (
    "chair|table|desk|couch|sofa|bed|door|wall|shelf|cabinet|television|tv"
    "|plant|lamp|counter|staircase|stairs|hallway|kitchen|bathroom|bedroom"
    "|living room|dining room|window|pillar|column|fireplace|bookcase"
    "|refrigerator|sink|toilet|bathtub|mirror|rug|carpet|painting|picture"
)
_LANDMARK_RE = re.compile(
    rf"\b(?:(?:{_LANDMARK_COLOURS})\s+)?(?:{_LANDMARK_OBJECTS})s?\b",
    re.IGNORECASE,
)


def _extract_landmarks(instruction: str) -> List[str]:
    found = _LANDMARK_RE.findall(instruction)
    return list({lm.lower().strip() for lm in found})


@dataclass
class _InferenceMemory:
    """Dead-reckoning spatial memory for real-robot deployment.

    Tracks position from accumulated forward / turn primitives and builds a
    text summary that is injected into every VLM prompt turn.
    """
    landmarks: List[str] = field(default_factory=list)
    visited_landmarks: List[str] = field(default_factory=list)

    # Dead-reckoning state (x metres East, y metres North, heading radians)
    _x: float = 0.0
    _y: float = 0.0
    _heading: float = 0.0  # 0 = North, +ve = left (CCW)

    # Approximate total distance travelled
    _dist_m: float = 0.0
    _steps: int = 0

    def reset(self):
        self._x = self._y = self._heading = self._dist_m = 0.0
        self._steps = 0
        self.visited_landmarks = []

    def update(self, action_id: int, numeric: float):
        """Update dead-reckoning from a single executed primitive."""
        self._steps += 1
        if action_id == 1:  # forward
            d = numeric / 100.0
            self._x += d * math.sin(self._heading)
            self._y += d * math.cos(self._heading)
            self._dist_m += d
        elif action_id == 2:  # turn left
            self._heading += math.radians(numeric)
        elif action_id == 3:  # turn right
            self._heading -= math.radians(numeric)

        # Landmark proximity proxy: evenly divide expected path among landmarks
        if self.landmarks:
            step_per_lm = max(1, 40 // len(self.landmarks))  # ~40 step budget
            for i, lm in enumerate(self.landmarks):
                if lm not in self.visited_landmarks and self._steps >= (i + 1) * step_per_lm:
                    self.visited_landmarks.append(lm)
                    print(f"[Memory] Landmark checkpoint: '{lm}'")

    def get_summary(self) -> str:
        if self._steps == 0:
            return ""
        heading_deg = math.degrees(self._heading) % 360
        lm_str = ""
        if self.visited_landmarks:
            lm_str = f" Confirmed: {', '.join(self.visited_landmarks)}."
        summary = (
            f"[Memory] {self._steps} steps, ~{self._dist_m:.1f}m travelled, "
            f"heading ~{heading_deg:.0f}°.{lm_str}"
        )
        # Persist to /tmp for web_ui to pick up
        try:
            with open(_MEMORY_FILE, "w") as f:
                json.dump({
                    "steps": self._steps,
                    "dist_m": round(self._dist_m, 2),
                    "heading_deg": round(heading_deg, 1),
                    "x": round(self._x, 3),
                    "y": round(self._y, 3),
                    "landmarks": self.landmarks,
                    "visited_landmarks": self.visited_landmarks,
                }, f)
        except Exception:
            pass
        return summary


# ── Depth-based obstacle detection ──────────────────────────────────────────

def _check_depth_obstacle(
    depth_frame: Optional[np.ndarray],
    threshold_m: float = 0.8,
    centre_fraction: float = 1 / 3,
) -> bool:
    """Return True if the central FOV crop has any depth reading below threshold.

    Args:
        depth_frame: (H, W) float32 array of depth in metres, or None.
        threshold_m: Distance in metres below which a pixel is considered blocked.
        centre_fraction: Fraction of H and W used for the central crop.
    """
    if depth_frame is None:
        return False
    h, w = depth_frame.shape[:2]
    margin_h = int(h * (1 - centre_fraction) / 2)
    margin_w = int(w * (1 - centre_fraction) / 2)
    crop = depth_frame[margin_h: h - margin_h, margin_w: w - margin_w]
    # Ignore zero / NaN (sensor dropout)
    valid = crop[(crop > 0.01) & np.isfinite(crop)]
    if valid.size == 0:
        return False
    return bool(valid.min() < threshold_m)

MAX_PIXELS = 76_800
PIXEL_FACTOR = 28

# ── action parsing patterns (strict, matches exact R2R/RxR format) ───────────
_THINK_RE  = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_FWD_RE    = re.compile(r'\bmove forward (25|50|75)cm\b',     re.IGNORECASE)
_LEFT_R2R  = re.compile(r'\bturn left (15|30|45) degrees\b',  re.IGNORECASE)
_RIGHT_R2R = re.compile(r'\bturn right (15|30|45) degrees\b', re.IGNORECASE)
_LEFT_RXR  = re.compile(r'\bturn left (30|60|90) degrees\b',  re.IGNORECASE)
_RIGHT_RXR = re.compile(r'\bturn right (30|60|90) degrees\b', re.IGNORECASE)


@dataclass(frozen=True)
class FrameSnapshot:
    seq: int = 0
    ts: float = 0.0
    signature: bytes = b""
    duplicate_streak: int = 0


def _frame_signature(bgr: np.ndarray) -> bytes:
    if bgr.size == 0:
        return b""
    sample = np.ascontiguousarray(bgr[::16, ::16])
    return hashlib.blake2s(sample.tobytes(), digest_size=8).digest()


# ── image helpers ────────────────────────────────────────────────────────────

def _smart_resize(h: int, w: int) -> Tuple[int, int]:
    factor = PIXEL_FACTOR
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((h * w) / MAX_PIXELS)
        h_bar = math.floor(h / beta / factor) * factor
        w_bar = math.floor(w / beta / factor) * factor
    return max(factor, h_bar), max(factor, w_bar)


def _encode_pil(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _crop_to_sim(bgr: np.ndarray) -> np.ndarray:
    """
    Resize to 640×480 to match StreamVLN / Habitat training resolution.
    No letterboxing — just direct resize, same as StreamVLN real-world deploy.
    """
    target_w, target_h = 640, 480
    h, w = bgr.shape[:2]
    if w == target_w and h == target_h:
        return bgr
    return cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    if cv2 is None or Image is None:
        raise RuntimeError("Camera/image support requires opencv-python and Pillow.")

    bgr = _crop_to_sim(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if _qwen_smart_resize is not None:
        rh, rw = _qwen_smart_resize(
            pil.height,
            pil.width,
            max_pixels=MAX_PIXELS,
            factor=PIXEL_FACTOR,
        )
    else:
        rh, rw = _smart_resize(pil.height, pil.width)
    return pil.resize((rw, rh), Image.LANCZOS)


def _wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _angle_delta_rad(current: float, start: float) -> float:
    return _wrap_angle_rad(current - start)


def parse_action_text(
    raw: str,
    action_space: str = "r2r",
    max_actions: int = 3,
) -> List[Tuple[int, float]]:
    """
    Parse model-style action text into primitive actions.
    Action encoding: 0=stop  1=forward  2=turn_left  3=turn_right
    """
    assert action_space in ("r2r", "rxr")
    text = _THINK_RE.sub('', raw).strip()
    text = re.sub(r'[\n;|]+', ',', text)
    left_re = _LEFT_R2R if action_space == "r2r" else _LEFT_RXR
    right_re = _RIGHT_R2R if action_space == "r2r" else _RIGHT_RXR

    out: List[Tuple[int, float]] = []
    for part in text.split(","):
        part = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', part).strip()
        if not part:
            continue
        if re.fullmatch(r'stop', part, re.IGNORECASE):
            out.append((0, 0.0))
        elif m := _FWD_RE.search(part):
            out.append((1, float(m.group(1))))
        elif m := left_re.search(part):
            out.append((2, float(m.group(1))))
        elif m := right_re.search(part):
            out.append((3, float(m.group(1))))
        if len(out) >= max_actions:
            break

    if not out:
        raise ValueError(f"No valid actions parsed from: {raw!r}")
    return out


# ── Diadem AGV controller (ros2 topic pub via subprocess) ────────────────────

class DiademDirectScriptController:
    """
    Drives the Diadem AGV using ros2 topic pub subprocess calls — identical
    approach to diadem/diadem.sh, confirmed working on hardware.
    """

    AGENT_BIN   = "/home/chitti/uros_ws/install/micro_ros_agent/lib/micro_ros_agent/micro_ros_agent"
    LINEAR_SPD  = 0.3   # m/s
    ANGULAR_SPD = 0.6   # rad/s
    RATE        = 10    # publish Hz

    def __init__(
        self,
        dev: str = "/dev/esp",
        dry_run: bool = False,
        agent_wait_s: float = 8.0,
        baud: int = 921600,
        ros_domain: int = 101,
        agent_log: str = "/tmp/agv_micro_ros_agent.log",
        estop_path: str = "/tmp/vln_estop.flag",
    ):
        self.dry_run = dry_run
        self._estop_path = estop_path
        self._agent: Optional[subprocess.Popen] = None
        self._env = os.environ.copy()
        # ros2 uses #!/usr/bin/python3 (system Python3); system numpy on this
        # Jetson is broken. Ensure conda's working numpy appears in PYTHONPATH
        # before the system site-packages so ros2 finds it first.
        self._env.pop("PYTHONHOME", None)
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            conda_sp = f"{conda_prefix}/lib/python3.10/site-packages"
            pp = self._env.get("PYTHONPATH", "")
            if conda_sp not in pp:
                self._env["PYTHONPATH"] = f"{pp}:{conda_sp}".strip(":")
        self._env["ROS_DOMAIN_ID"] = str(ros_domain)

        if dry_run:
            print("[Diadem] DRY-RUN — commands logged, not executed")
            return

        # Start agent if not already running
        if subprocess.run(["pgrep", "-f", "micro_ros_agent"],
                          capture_output=True).returncode != 0:
            subprocess.run(["pkill", "-9", "-f", "micro_ros_agent"], check=False)
            time.sleep(1)
            print(f"[Diadem] Starting micro_ros_agent on {dev} @ {baud} baud…")
            log_f = open(agent_log, "a")
            self._agent = subprocess.Popen(
                [self.AGENT_BIN, "serial", "--dev", dev, "-b", str(baud)],
                stdout=log_f, stderr=log_f, env=self._env,
            )
            print(f"[Diadem] Waiting {agent_wait_s:.0f}s for ESP32 to connect…")
            time.sleep(agent_wait_s)
        else:
            print("[Diadem] micro_ros_agent already running.")

        # Wait for /cmd_vel subscriber (ESP32)
        print("[Diadem] Waiting for /cmd_vel subscriber…")
        for i in range(15):
            out = subprocess.run(
                ["ros2", "topic", "info", "/cmd_vel"],
                capture_output=True, text=True, env=self._env,
            ).stdout
            for line in out.splitlines():
                if "Subscription count:" in line and int(line.split()[-1]) > 0:
                    print("[Diadem] ESP32 ready — /cmd_vel active.")
                    return
            time.sleep(1)
        print("[Diadem] WARNING: no subscriber on /cmd_vel after 15s — proceeding anyway")

    def _pub(self, linear: float, angular: float, times: int):
        if self._estop_active():
            return
        msg = (f"{{linear: {{x: {linear}, y: 0.0, z: 0.0}}, "
               f"angular: {{x: 0.0, y: 0.0, z: {angular}}}}}")
        subprocess.run(
            ["ros2", "topic", "pub",
             "--times", str(times), "--rate", str(self.RATE),
             "/cmd_vel", "geometry_msgs/msg/Twist", msg],
            env=self._env, check=False,
        )

    def _stop(self):
        subprocess.run(
            ["ros2", "topic", "pub", "--once",
             "/cmd_vel", "geometry_msgs/msg/Twist",
             "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"],
            env=self._env, check=False,
        )

    def _estop_active(self) -> bool:
        return os.path.exists(self._estop_path)

    def move_forward(self, dist_cm: float):
        print(f"[Diadem] Forward {dist_cm:.0f}cm")
        if self.dry_run:
            return
        dist_m = dist_cm / 100.0
        speed = self.LINEAR_SPD if dist_m > 0 else -self.LINEAR_SPD
        times = max(1, round(abs(dist_m) / self.LINEAR_SPD * self.RATE))
        self._pub(speed, 0.0, times)
        self._stop()

    def turn_left(self, deg: float):
        print(f"[Diadem] Turn left {deg:.0f}°")
        if self.dry_run:
            return
        times = max(1, round(math.radians(abs(deg)) / self.ANGULAR_SPD * self.RATE))
        self._pub(0.0, self.ANGULAR_SPD, times)
        self._stop()

    def turn_right(self, deg: float):
        print(f"[Diadem] Turn right {deg:.0f}°")
        if self.dry_run:
            return
        times = max(1, round(math.radians(abs(deg)) / self.ANGULAR_SPD * self.RATE))
        self._pub(0.0, -self.ANGULAR_SPD, times)
        self._stop()

    def stop(self):
        print("[Diadem] STOP")
        if self.dry_run:
            return
        self._stop()

    def close(self):
        if self.dry_run:
            return
        try:
            self._stop()
        except Exception:
            pass
        if self._agent is not None:
            try:
                self._agent.terminate()
            except Exception:
                pass
            self._agent = None
        print("[Diadem] Closed.")


# ── Go1 controller ───────────────────────────────────────────────────────────

class Go1Controller:
    """
    Drives the Unitree Go1 via HighCmd UDP (port 8082 on robot, 8090 local).
    Uses Unitree HighState odometry/IMU feedback to stop each primitive.
    """

    HIGHLEVEL  = 0xee
    ROBOT_IP   = "192.168.123.161"
    LOCAL_PORT = 8090
    ROBOT_PORT = 8082

    VX_FWD  = 0.12  # conservative forward speed for odom-based stopping
    WZ_TURN = 0.18  # conservative turn speed for odom/imu-based stopping

    CTRL_DT = 0.002  # 500 Hz control loop
    STOP_HOLD_S = 0.60
    CMD_GUARD_S = 0.10
    DIST_TOL_M = 0.03
    YAW_TOL_RAD = math.radians(4.0)
    MAX_PRIMITIVE_S = 8.0

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._udp = self._cmd = self._state = None

        if not dry_run:
            sys.path.insert(0, _SDK_LIB)
            import robot_interface as sdk
            self._udp   = sdk.UDP(self.HIGHLEVEL, self.LOCAL_PORT,
                                  self.ROBOT_IP, self.ROBOT_PORT)
            self._cmd   = sdk.HighCmd()
            self._state = sdk.HighState()
            self._udp.InitCmdData(self._cmd)
            print("[Go1] UDP connected.")

    def _estop_active(self) -> bool:
        return os.path.exists(_E_STOP_FLAG)

    def _make_cmd(self, vx: float, wz: float, moving: bool):
        cmd = self._cmd
        cmd.mode            = 2 if moving else 1
        cmd.gaitType        = 1 if moving else 0
        cmd.velocity        = [vx, 0.0]
        cmd.yawSpeed        = wz
        cmd.footRaiseHeight = 0.08 if moving else 0.0
        cmd.bodyHeight      = 0.0
        cmd.euler           = [0.0, 0.0, 0.0]
        cmd.speedLevel      = 0
        cmd.reserve         = 0

    def _update_state(self) -> bool:
        if self.dry_run or self._udp is None or self._state is None:
            return False
        try:
            self._udp.Recv()
            self._udp.GetRecv(self._state)
            return True
        except Exception:
            return False

    def _current_xy_yaw(self) -> Optional[Tuple[float, float, float]]:
        if not self._update_state():
            return None
        try:
            x = float(self._state.position[0])
            y = float(self._state.position[1])
            yaw = float(self._state.imu.rpy[2])
            return x, y, yaw
        except Exception:
            return None

    def _hold_stop(self, duration: float = STOP_HOLD_S):
        if self.dry_run or self._udp is None:
            if duration > 0:
                time.sleep(duration)
            return
        deadline = time.time() + max(0.0, duration)
        while time.time() < deadline:
            self._update_state()
            self._make_cmd(0.0, 0.0, moving=False)
            self._udp.SetSend(self._cmd)
            self._udp.Send()
            time.sleep(self.CTRL_DT)

    def _send_until(self, vx: float, wz: float, done_fn, timeout_s: float):
        if self.dry_run or self._udp is None:
            print(f"  [DRY] vx={vx:+.2f} wz={wz:+.2f} t<={timeout_s:.2f}s")
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                if self._estop_active():
                    print("  [DRY] interrupted by EMERGENCY STOP")
                    return
                if done_fn(None):
                    break
                time.sleep(min(0.05, deadline - time.time()))
            return
        deadline = time.time() + max(self.CTRL_DT, timeout_s)
        started = time.time()
        while time.time() < deadline:
            if self._estop_active():
                print("[Go1] Motion interrupted by EMERGENCY STOP")
                break
            self._update_state()
            self._make_cmd(vx, wz, moving=True)
            self._udp.SetSend(self._cmd)
            self._udp.Send()
            if time.time() - started >= self.CMD_GUARD_S and done_fn(self._state):
                break
            time.sleep(self.CTRL_DT)
        self._hold_stop()

    def move_forward(self, dist_cm: float):
        target_m = dist_cm / 100.0
        timeout_s = min(self.MAX_PRIMITIVE_S, max(1.2, target_m / max(0.03, self.VX_FWD) + 1.5))
        start_pose = self._current_xy_yaw()
        print(f"[Go1] Forward {dist_cm:.0f}cm  (closed-loop, timeout {timeout_s:.2f}s)")

        if start_pose is None:
            raise RuntimeError("Go1 odometry unavailable for forward motion.")

        x0, y0, _yaw0 = start_pose

        def done_fn(state) -> bool:
            if state is None:
                return False
            try:
                dx = float(state.position[0]) - x0
                dy = float(state.position[1]) - y0
            except Exception:
                return False
            return math.hypot(dx, dy) >= max(0.0, target_m - self.DIST_TOL_M)

        self._send_until(self.VX_FWD, 0.0, done_fn, timeout_s)

    def turn_left(self, deg: float):
        target_rad = math.radians(deg)
        timeout_s = min(self.MAX_PRIMITIVE_S, max(1.0, target_rad / max(0.05, self.WZ_TURN) + 1.2))
        start_pose = self._current_xy_yaw()
        print(f"[Go1] Turn left  {deg:.0f}°  (closed-loop, timeout {timeout_s:.2f}s)")

        if start_pose is None:
            raise RuntimeError("Go1 IMU/odometry unavailable for left turn.")

        _, _, yaw0 = start_pose

        def done_fn(state) -> bool:
            if state is None:
                return False
            try:
                yaw = float(state.imu.rpy[2])
            except Exception:
                return False
            return _angle_delta_rad(yaw, yaw0) >= max(0.0, target_rad - self.YAW_TOL_RAD)

        self._send_until(0.0, +self.WZ_TURN, done_fn, timeout_s)

    def turn_right(self, deg: float):
        target_rad = math.radians(deg)
        timeout_s = min(self.MAX_PRIMITIVE_S, max(1.0, target_rad / max(0.05, self.WZ_TURN) + 1.2))
        start_pose = self._current_xy_yaw()
        print(f"[Go1] Turn right {deg:.0f}°  (closed-loop, timeout {timeout_s:.2f}s)")

        if start_pose is None:
            raise RuntimeError("Go1 IMU/odometry unavailable for right turn.")

        _, _, yaw0 = start_pose

        def done_fn(state) -> bool:
            if state is None:
                return False
            try:
                yaw = float(state.imu.rpy[2])
            except Exception:
                return False
            return -_angle_delta_rad(yaw, yaw0) >= max(0.0, target_rad - self.YAW_TOL_RAD)

        self._send_until(0.0, -self.WZ_TURN, done_fn, timeout_s)

    def stop(self):
        print("[Go1] STOP")
        if not self.dry_run and self._udp is not None:
            self._hold_stop()
        else:
            print("  [DRY] stop")


# ── VLN agent (queries checkpoint server) ───────────────────────────────────

class ActiveVLNAgent:
    """
    Queries the vLLM checkpoint server (OpenAI-compatible API at --base-url)
    with a multi-turn conversation and returns primitive Go1 actions.
    Action encoding: 0=stop  1=forward  2=turn_left  3=turn_right

    Extended capabilities:
      - Spatial memory: dead-reckoning map injected as text into each VLM prompt.
      - Closed-loop replanning: uses "[Replanning — obstacle detected]" prefix
        when the depth sensor detected a nearby obstacle after the last action.
      - Semantic landmark tracking: instruction landmarks logged and included in
        the memory summary as the agent progresses.
    """

    def __init__(
        self,
        base_url             : str,
        api_key              : str   = "EMPTY",
        action_space         : str   = "r2r",
        max_turns            : int   = 40,
        max_steps            : int   = 120,
        enable_memory        : bool  = False,
        enable_landmark_tracking: bool = False,
    ):
        assert action_space in ("r2r", "rxr")
        self.action_space = action_space
        self.max_turns    = max_turns
        self.max_steps    = max_steps
        self.turn_step    = 15 if action_space == "r2r" else 30

        self.enable_memory            = enable_memory
        self.enable_landmark_tracking = enable_landmark_tracking

        if OpenAI is None:
            raise RuntimeError("OpenAI Python package is required for VLN inference.")
        self.client     = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = self.client.models.list().data[0].id
        print(f"[VLN] Connected to server — model={self.model_name}")

        self._memory = _InferenceMemory()
        self._reset()

    # ── state ────────────────────────────────────────────────────────────────

    def _reset(self):
        sys_prompt = SYSTEM_PROMPT_R2R if self.action_space == "r2r" else SYSTEM_PROMPT_RXR
        self._conv: List[dict] = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]}
        ]
        self._turn_count = 0
        self._step_count = 0
        self._done       = False
        self._memory.reset()

    def reset(self):
        self._reset()

    def set_instruction(self, instruction: str):
        """Call once per episode to initialise landmark extraction."""
        if self.enable_landmark_tracking:
            self._memory.landmarks = _extract_landmarks(instruction)
            if self._memory.landmarks:
                print(f"[Memory] Landmarks detected: {self._memory.landmarks}")

    def update_position(self, action_id: int, numeric: float):
        """Update dead-reckoning memory after a physical primitive is executed."""
        if self.enable_memory or self.enable_landmark_tracking:
            self._memory.update(action_id, numeric)

    # ── inference ────────────────────────────────────────────────────────────

    def _call_server(self) -> str:
        resp = self.client.chat.completions.create(
            model                 = self.model_name,
            messages              = self._conv,
            max_completion_tokens = 512,
            temperature           = 0.2,
            top_p                 = 0.8,
        )
        return resp.choices[0].message.content.strip()

    # ── action parsing (strict regex matching exact R2R/RxR action format) ────

    def _expand(self, raw: str) -> List[Tuple[int, float]]:
        """Parse comma-separated model output into actions using exact R2R/RxR patterns."""
        return parse_action_text(
            raw,
            action_space=self.action_space,
            max_actions=3,
        )

    # ── public API ───────────────────────────────────────────────────────────

    def step(
        self,
        bgr: np.ndarray,
        instruction: str,
        replan: bool = False,
    ) -> Tuple[List[Tuple[int, float]], str, Optional["Image.Image"]]:
        """Run one VLM inference call and return all actions for this step.

        Args:
            bgr: Current BGR camera frame.
            instruction: Navigation instruction text.
            replan: When True the prompt prefix signals an obstacle-triggered
                    re-inference rather than a normal post-action observation.

        Returns:
            ([(action_id, numeric), ...], raw_text, model_pil)
            model_pil is the smart-resized PIL image sent to the model.
            Terminal returns use [(0, 0.0)].
        """
        if self._done:
            return [(0, 0.0)], "done", None

        self._turn_count += 1
        if self._turn_count > self.max_turns:
            self._done = True
            return [(0, 0.0)], "max_turns", None

        if self._step_count >= self.max_steps:
            self._done = True
            return [(0, 0.0)], "max_steps", None

        pil = _bgr_to_pil(bgr)

        if self._turn_count == 1:
            prefix = "[Initial Observation]:"
        elif replan:
            prefix = "[Replanning — obstacle detected]:"
            # Write flag file for web_ui to display obstacle alert
            try:
                open(_OBSTACLE_FILE, "w").write("1")
            except Exception:
                pass
        else:
            prefix = "After that, the observation is:"
            # Clear obstacle flag on normal step
            try:
                os.remove(_OBSTACLE_FILE)
            except OSError:
                pass

        memory_text = ""
        if self.enable_memory or self.enable_landmark_tracking:
            summary = self._memory.get_summary()
            if summary:
                memory_text = summary + "\n"

        self._conv.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prefix},
                {"type": "image_url", "image_url": {"url": _encode_pil(pil)}},
                {"type": "text", "text": USER_PROMPT.format(
                    instruction=instruction,
                    memory=memory_text,
                )},
            ],
        })

        try:
            raw = self._call_server()
        except Exception as exc:
            print(f"[VLN] Inference error: {exc}")
            self._done = True
            return [(0, 0.0)], "error", pil

        self._conv.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})

        primitives = self._expand(raw)

        # Enforce max_steps budget — don't return more actions than remaining budget
        budget = self.max_steps - self._step_count
        primitives = primitives[:max(1, budget)]

        # Truncate after the first STOP — nothing should execute after a stop
        truncated: List[Tuple[int, float]] = []
        for a in primitives:
            truncated.append(a)
            if a[0] == 0:
                break
        primitives = truncated

        self._step_count += len(primitives)
        if primitives[-1][0] == 0:
            self._done = True

        _lbl = {0: "STOP", 1: "FWD", 2: "LEFT", 3: "RIGHT"}
        print(f"[VLN] raw='{raw}'  →  parsed={[(_lbl.get(a, a), v) for a, v in primitives]}")
        return primitives, raw, pil


# ── thread-safe camera ───────────────────────────────────────────────────────

class _FrameBufferMixin:
    SHARED_FRAME = "/tmp/vln_latest_frame.jpg"
    DUPLICATE_STREAK_LIMIT = 90

    def _init_frame_buffer(self):
        self._frame: Optional[np.ndarray] = None
        self._frame_ts = 0.0
        self._frame_seq = 0
        self._frame_sig = b""
        self._duplicate_streak = 0
        self._lock = threading.Lock()
        self._restart_reason = ""

    def _publish_frame(self, frame: np.ndarray):
        signature = _frame_signature(frame)
        now = time.monotonic()
        with self._lock:
            duplicate_streak = self._duplicate_streak + 1 if signature == self._frame_sig else 0
            self._frame = frame
            self._frame_ts = now
            self._frame_seq += 1
            self._frame_sig = signature
            self._duplicate_streak = duplicate_streak
        try:
            cv2.imwrite(self.SHARED_FRAME, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception:
            pass

    def snapshot(self) -> FrameSnapshot:
        with self._lock:
            return FrameSnapshot(
                seq=self._frame_seq,
                ts=self._frame_ts,
                signature=self._frame_sig,
                duplicate_streak=self._duplicate_streak,
            )

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def read_with_snapshot(self) -> Tuple[Optional[np.ndarray], FrameSnapshot]:
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            snap = FrameSnapshot(
                seq=self._frame_seq,
                ts=self._frame_ts,
                signature=self._frame_sig,
                duplicate_streak=self._duplicate_streak,
            )
        return frame, snap

    def wait_for_frame(
        self,
        after: Optional[FrameSnapshot] = None,
        timeout_s: float = 4.0,
        require_new_content: bool = False,
    ) -> Tuple[Optional[np.ndarray], FrameSnapshot]:
        deadline = time.monotonic() + max(0.1, timeout_s)
        last_snap = self.snapshot()
        while time.monotonic() < deadline:
            frame, snap = self.read_with_snapshot()
            last_snap = snap
            if frame is None:
                time.sleep(0.02)
                continue
            if after is None or snap.seq > after.seq:
                if not require_new_content:
                    return frame, snap
                if not after or not after.signature or snap.signature != after.signature:
                    return frame, snap
                if snap.duplicate_streak >= self.DUPLICATE_STREAK_LIMIT:
                    return None, snap
            time.sleep(0.01)
        return None, last_snap

    def restart(self, reason: str = ""):
        self._restart_reason = reason
        self._request_restart()


class Camera(_FrameBufferMixin):
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        if cv2 is None:
            raise RuntimeError("Camera support requires opencv-python.")
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._cap: Optional["cv2.VideoCapture"] = None
        self._restart_event = threading.Event()
        self._stopped = False
        self._init_frame_buffer()
        self._open_capture(initial=True)
        threading.Thread(target=self._loop, daemon=True).start()
        time.sleep(0.6)

    def _open_capture(self, initial: bool = False):
        cap = cv2.VideoCapture(self._camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera {self._camera_id}")
        old_cap, self._cap = self._cap, cap
        if old_cap is not None:
            old_cap.release()
        action = "opened" if initial else "reopened"
        print(f"[Camera] Webcam {self._camera_id} {action} at {self._width}x{self._height}")

    def _request_restart(self):
        self._restart_event.set()

    def _loop(self):
        read_failures = 0
        while not self._stopped:
            if self._restart_event.is_set():
                reason = self._restart_reason or "restart requested"
                self._restart_event.clear()
                try:
                    print(f"[Camera] Restarting webcam {self._camera_id}: {reason}")
                    self._open_capture()
                except Exception as exc:
                    print(f"[Camera] Webcam restart failed: {exc}")
                    time.sleep(0.5)
                    continue

            cap = self._cap
            if cap is None:
                time.sleep(0.05)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                read_failures += 1
                if read_failures >= 10:
                    self.restart("webcam read failed repeatedly")
                    read_failures = 0
                time.sleep(0.05)
                continue

            read_failures = 0
            self._publish_frame(frame)

            if self.snapshot().duplicate_streak >= self.DUPLICATE_STREAK_LIMIT:
                self.restart("webcam feed repeated identical frames")

    def release(self):
        self._stopped = True
        if self._cap is not None:
            self._cap.release()


class RealSenseCamera(_FrameBufferMixin):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
    ):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense support requires pyrealsense2. Install librealsense/pyrealsense2 and rerun with --rs."
            ) from exc

        self._rs = rs
        self._width = width
        self._height = height
        self._fps = fps
        self._enable_depth = enable_depth
        self._pipeline = None
        self._restart_event = threading.Event()
        self._stopped = False
        self._init_frame_buffer()
        # Depth frame — only populated when enable_depth=True
        self._depth_frame: Optional[np.ndarray] = None
        self._depth_lock = threading.Lock()
        self._open_pipeline(initial=True)
        threading.Thread(target=self._loop, daemon=True).start()
        time.sleep(0.6)

    def read_depth(self) -> Optional[np.ndarray]:
        """Return the most recent depth frame (H×W float32, metres), or None."""
        with self._depth_lock:
            return self._depth_frame.copy() if self._depth_frame is not None else None

    def _open_pipeline(self, initial: bool = False):
        rs = self._rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        if self._enable_depth:
            config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
        try:
            pipeline.start(config)
        except Exception as exc:
            raise RuntimeError(f"Cannot start Intel RealSense color stream: {exc}") from exc

        # Match ROS2 realsense2_camera defaults used by StreamVLN real-world deploy
        try:
            device = pipeline.get_active_profile().get_device()
            color_sensor = device.first_color_sensor()
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
            if color_sensor.supports(rs.option.backlight_compensation):
                color_sensor.set_option(rs.option.backlight_compensation, 0)
            if color_sensor.supports(rs.option.power_line_frequency):
                color_sensor.set_option(rs.option.power_line_frequency, 2)  # 60 Hz
            print(f"[Camera] RealSense sensor options configured (auto-exposure, auto-WB)")
        except Exception as exc:
            print(f"[Camera] Warning: could not configure sensor options: {exc}")

        # Discard initial frames so auto-exposure/WB can settle
        for _ in range(30):
            try:
                pipeline.wait_for_frames(timeout_ms=500)
            except Exception:
                break

        old_pipeline, self._pipeline = self._pipeline, pipeline
        if old_pipeline is not None:
            try:
                old_pipeline.stop()
            except Exception:
                pass
        action = "started" if initial else "restarted"
        print(f"[Camera] RealSense {action} at {self._width}x{self._height}@{self._fps}")

    def _request_restart(self):
        self._restart_event.set()

    def _loop(self):
        read_failures = 0
        while not self._stopped:
            if self._restart_event.is_set():
                reason = self._restart_reason or "restart requested"
                self._restart_event.clear()
                try:
                    print(f"[Camera] Restarting RealSense stream: {reason}")
                    self._open_pipeline()
                except Exception as exc:
                    print(f"[Camera] RealSense restart failed: {exc}")
                    time.sleep(0.5)
                    continue

            pipeline = self._pipeline
            if pipeline is None:
                time.sleep(0.05)
                continue

            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    read_failures += 1
                    continue
                frame = np.asanyarray(color_frame.get_data()).copy()
                if self._enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_np = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001
                        with self._depth_lock:
                            self._depth_frame = depth_np
            except Exception:
                read_failures += 1
                if read_failures >= 5:
                    self.restart("RealSense stream timed out")
                    read_failures = 0
                time.sleep(0.05)
                continue

            read_failures = 0
            self._publish_frame(frame)

            if self.snapshot().duplicate_streak >= self.DUPLICATE_STREAK_LIMIT:
                self.restart("RealSense feed repeated identical frames")

    def release(self):
        self._stopped = True
        try:
            if self._pipeline is not None:
                self._pipeline.stop()
        except Exception:
            pass


# ── vLLM server helpers ──────────────────────────────────────────────────────

def _start_vllm(model_path: str, port: int, log_path: str,
                gpu_mem: float = 0.60, max_model_len: int = 32768) -> subprocess.Popen:
    """Launch the vLLM checkpoint server matching official ActiveVLN eval config."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--task", "generate",
        "--dtype", "auto",
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--allowed-origins", '["*"]',
        "--disable-log-requests",
        "--enable-prefix-caching",
        "--enforce-eager",            # needed on Jetson (no torch.compile)
        "--no-enable-chunked-prefill",
        "--limit-mm-per-prompt", '{"image": 200, "video": 0}',
        "--mm-processor-kwargs", '{"max_pixels": 76800}',
    ]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Use conda env lib dir for libcudnn/libcufile — same as infer.sh
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    # CUDA headers for Jetson JIT compilation
    _cuda_home = "/usr/local/cuda-12.6"
    env["CUDA_HOME"] = _cuda_home
    env["CUDA_PATH"] = _cuda_home
    env["CPATH"] = f"{_cuda_home}/targets/aarch64-linux/include:{env.get('CPATH', '')}"
    env["PATH"]  = f"{_cuda_home}/bin:{env.get('PATH', '')}"
    env.setdefault("TRITON_PTXAS_PATH", f"{_cuda_home}/bin/ptxas")
    log_f = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f,
                            start_new_session=True, env=env)
    print(f"[vLLM] PID={proc.pid}  model={model_path}  port={port}  log={log_path}")
    return proc


def _wait_vllm(port: int, timeout: int = 180) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    print(f"[vLLM] Waiting for server on port {port} (up to {timeout}s)…")
    for i in range(timeout):
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    print(f"[vLLM] Ready after {i+1}s.")
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ── main navigation loop ─────────────────────────────────────────────────────

def navigate(args):
    def _normalize_action(action_id: int, numeric: float) -> Tuple[int, float]:
        if not args.test_mode:
            return action_id, numeric
        if action_id == 1:
            return action_id, 25.0
        if action_id in (2, 3) and numeric > 15.0:
            return action_id, 30.0
        return action_id, numeric

    vllm_proc: Optional[subprocess.Popen] = None

    # Auto-launch vLLM server if requested
    if args.use_vllm and not args.base_url:
        vllm_proc = _start_vllm(
            model_path    = args.checkpoint_path,
            port          = args.vllm_port,
            log_path      = args.vllm_log,
            gpu_mem       = args.vllm_gpu_memory_utilization,
            max_model_len = args.vllm_max_model_len,
        )
        if not _wait_vllm(args.vllm_port, timeout=360):
            vllm_proc.terminate()
            raise RuntimeError("vLLM server did not become ready.")
        args.base_url = f"http://127.0.0.1:{args.vllm_port}/v1"
    elif not args.base_url:
        args.base_url = f"http://127.0.0.1:{args.vllm_port}/v1"

    agent = ActiveVLNAgent(
        base_url                 = args.base_url,
        api_key                  = args.api_key,
        action_space             = args.action_space,
        max_turns                = args.max_turns,
        max_steps                = args.max_steps,
        enable_memory            = args.enable_memory,
        enable_landmark_tracking = args.enable_landmark_tracking,
    )
    if args.diadem:
        robot = DiademDirectScriptController(
            dev=args.diadem_dev,
            dry_run=args.dry_run,
            agent_wait_s=args.diadem_agent_wait,
            baud=args.diadem_baud,
            ros_domain=args.diadem_ros_domain,
            agent_log=args.diadem_agent_log,
            estop_path=_E_STOP_FLAG,
        )
    else:
        robot = Go1Controller(dry_run=args.dry_run)

    use_camera = args.image_path is None
    cam: Optional[object] = None
    if use_camera:
        if args.rs:
            cam = RealSenseCamera(
                width=args.rs_width,
                height=args.rs_height,
                fps=args.rs_fps,
                enable_depth=args.enable_obstacle_replanning,
            )
        else:
            cam = Camera(camera_id=args.camera_id)

    _instr_file   = "/tmp/vln_instruction.txt"
    _pause_flag   = "/tmp/vln_paused.flag"
    _restart_flag = "/tmp/vln_restart.flag"
    _estop_flag   = _E_STOP_FLAG

    # Start paused if no instruction given — wait for UI
    instruction = args.instruction or ""
    if not instruction:
        open(_pause_flag, 'w').write("1")
        print("[NAV] *** PAUSED *** — set an instruction in the Web UI to begin.")
    else:
        open(_instr_file, 'w').write(instruction)
        agent.set_instruction(instruction)
    next_frame_ref = cam.snapshot() if use_camera and instruction and hasattr(cam, "snapshot") else None

    print(f"\n{'='*60}")
    print(f"  Instruction       : {instruction or '(waiting for UI)'}")
    print(f"  Action space      : {args.action_space}")
    print(f"  Server            : {args.base_url}")
    print(f"  Robot             : {'Diadem AGV direct USB' if args.diadem else 'Unitree Go1'}")
    print(f"  Max turns         : {args.max_turns}   Max steps: {args.max_steps}")
    print(f"  Dry-run           : {args.dry_run}")
    print(f"  RGB input         : {'realsense' if args.rs else ('image file' if args.image_path else f'webcam:{args.camera_id}')}")
    print(f"  Obstacle replan   : {args.enable_obstacle_replanning}"
          + (f" (threshold {args.obstacle_depth_threshold}m)" if args.enable_obstacle_replanning else ""))
    print(f"  Spatial memory    : {args.enable_memory}")
    print(f"  Landmark tracking : {args.enable_landmark_tracking}")
    print(f"{'='*60}\n")

    if args.test_mode:
        print("[NAV] Test mode enabled — clamping FORWARD to 25cm and TURN >15deg to 30deg")

    step = 0
    _obs_idx = 0
    _was_paused = not bool(instruction)
    _was_estopped = False
    _replan_next = False          # set True after obstacle detected
    _clear_obs_history()
    try:
        os.remove(_OBSTACLE_FILE)
    except OSError:
        pass

    try:
        while True:
            if os.path.exists(_estop_flag):
                if not _was_estopped:
                    robot.stop()
                    open(_pause_flag, 'w').write("1")
                    print("[NAV] *** EMERGENCY STOP *** current and future actions halted")
                    _was_estopped = True
                    _was_paused = True
                time.sleep(0.1)
                continue
            elif _was_estopped:
                print("[NAV] *** EMERGENCY STOP CLEARED ***")
                _was_estopped = False

            # ── pause gate ───────────────────────────────────────
            if os.path.exists(_pause_flag):
                if not _was_paused:
                    print("[NAV] *** PAUSED *** waiting for new instruction…")
                    _was_paused = True
                time.sleep(0.4)
                continue
            elif _was_paused:
                _was_paused = False
                print("[NAV] *** RESUMED ***")

            # ── restart flag ─────────────────────────────────────
            if os.path.exists(_restart_flag):
                try:
                    new_instr = open(_restart_flag).read().strip()
                    os.remove(_restart_flag)
                    agent.reset()
                    step = 0
                    _replan_next = False
                    _clear_obs_history(); _obs_idx = 0
                    if new_instr:
                        instruction = new_instr
                        open(_instr_file, 'w').write(instruction)
                        agent.set_instruction(instruction)
                        next_frame_ref = cam.snapshot() if use_camera and hasattr(cam, "snapshot") else None
                        print(f"[NAV] *** RESTART *** instruction={instruction}")
                    else:
                        open(_pause_flag, 'w').write("1")
                        _was_paused = True
                        print("[NAV] *** RESTART *** pausing for new instruction")
                        continue
                except Exception:
                    pass

            # ── live instruction update ──────────────────────────
            if os.path.exists(_instr_file):
                try:
                    live = open(_instr_file).read().strip()
                    if live and live != instruction:
                        print(f"[NAV] *** NEW INSTRUCTION *** → {live}")
                        instruction = live
                        next_frame_ref = cam.snapshot() if use_camera and hasattr(cam, "snapshot") else None
                        agent.reset()
                        agent.set_instruction(instruction)
                        step = 0
                        _replan_next = False
                        _clear_obs_history(); _obs_idx = 0
                except Exception:
                    pass

            if not instruction:
                time.sleep(0.4)
                continue

            # ── grab frame ───────────────────────────────────────
            if use_camera:
                # In dry-run with a live camera, leave time for manual camera
                # repositioning so the next observation matches the prior action.
                # Skip the settle wait on step 0 — no prior action to settle from.
                wait_s = 5.0 if args.dry_run else (args.settle_time if step > 0 else 0.0)
                if wait_s > 0:
                    if args.dry_run:
                        print(f"[NAV] Waiting {wait_s:.1f}s for manual camera reposition…")
                    time.sleep(wait_s)
                if next_frame_ref is not None and hasattr(cam, "wait_for_frame"):
                    frame, frame_snap = cam.wait_for_frame(
                        after=next_frame_ref,
                        timeout_s=args.frame_timeout,
                        require_new_content=step > 0,
                    )
                    duplicate_limit = getattr(cam, "DUPLICATE_STREAK_LIMIT", 90)
                    if frame is None or frame_snap.duplicate_streak >= duplicate_limit:
                        print("[NAV] Camera did not produce a fresh frame; restarting camera stream.")
                        if hasattr(cam, "restart"):
                            cam.restart("navigation loop detected stale frame")
                        next_frame_ref = cam.snapshot() if hasattr(cam, "snapshot") else None
                        time.sleep(0.5)
                        continue
                    next_frame_ref = None
                else:
                    frame = cam.read()
                if frame is None:
                    time.sleep(0.05)
                    continue
            else:
                frame = cv2.imread(args.image_path)
                if frame is None:
                    raise FileNotFoundError(args.image_path)

            # Crop to simulation FOV (640×480, 4:3) — same view the model receives
            display_frame = _crop_to_sim(frame) if cv2 is not None else frame

            # Overwrite shared JPEG so the web UI live feed also shows the cropped view
            if cv2 is not None:
                try:
                    cv2.imwrite(Camera.SHARED_FRAME, display_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
                except Exception:
                    pass

            if not args.headless:
                try:
                    cv2.imshow("ActiveVLN — Go1", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[NAV] Quit by user.")
                        break
                except cv2.error:
                    args.headless = True

            # ── inference (pass replan flag if obstacle was detected) ─────────
            t0 = time.time()
            if _replan_next:
                print("[NAV] *** OBSTACLE REPLANNING *** triggering fresh VLM inference")
            actions, raw, model_pil = agent.step(display_frame, instruction, replan=_replan_next)
            _replan_next = False  # consumed
            actions = [_normalize_action(action_id, numeric) for action_id, numeric in actions]
            elapsed = time.time() - t0
            labels = {
                0: lambda v: "STOP",
                1: lambda v: f"FORWARD {v:.0f}cm",
                2: lambda v: f"TURN LEFT {v:.0f}°",
                3: lambda v: f"TURN RIGHT {v:.0f}°",
            }
            batch_txt = " → ".join(labels.get(aid, lambda _v: "?")(val) for aid, val in actions) or "STOP"
            mode_txt = "manual dry-run execution" if args.dry_run else "robot action execution"
            print(f"[NAV] Inference complete — waiting for {mode_txt}: {batch_txt}")
            # Save the exact smart-resized image the model received, not the raw camera frame
            if model_pil is not None and cv2 is not None:
                obs_bgr = cv2.cvtColor(np.array(model_pil), cv2.COLOR_RGB2BGR)
            else:
                obs_bgr = display_frame
            _append_obs(_obs_idx, obs_bgr, actions, raw)
            _obs_idx += 1

            # Execute ALL actions returned by this inference call before
            # looping back to the top — avoids instruction-check races.
            stop_issued = False
            for i, (action_id, numeric) in enumerate(actions):
                step += 1
                label = labels.get(action_id, lambda _v: "?")(numeric)
                timing = f"{elapsed:.1f}s  " if i == 0 else "       "
                print(f"[Step {step:03d}] {timing}→  {label}"
                      + (f"  ('{raw}')" if raw and i == 0 else ""))

                if os.path.exists(_estop_flag):
                    robot.stop()
                    open(_pause_flag, 'w').write("1")
                    print("[NAV] *** EMERGENCY STOP *** current and future actions halted")
                    stop_issued = True
                    _was_paused = True
                    _was_estopped = True
                    break
                if action_id == 0:
                    robot.stop()
                    print("\n[NAV] Navigation complete — waiting for next instruction…")
                    agent.reset()
                    step = 0
                    _replan_next = False
                    instruction = ""
                    open(_pause_flag, 'w').write("1")
                    _was_paused = True
                    print("[NAV] *** PAUSED *** waiting for new instruction…")
                    stop_issued = True
                    break
                elif action_id == 1:
                    robot.move_forward(numeric)
                elif action_id == 2:
                    robot.turn_left(numeric)
                elif action_id == 3:
                    robot.turn_right(numeric)

                # Update dead-reckoning memory after physical execution
                agent.update_position(action_id, numeric)

                # ── Closed-loop obstacle detection ────────────────
                if args.enable_obstacle_replanning and hasattr(cam, "read_depth"):
                    depth = cam.read_depth()
                    if _check_depth_obstacle(depth, threshold_m=args.obstacle_depth_threshold):
                        print(
                            f"[Obstacle] Depth < {args.obstacle_depth_threshold}m in central FOV — "
                            "scheduling replan on next turn."
                        )
                        _replan_next = True

                if os.path.exists(_estop_flag):
                    robot.stop()
                    open(_pause_flag, 'w').write("1")
                    print("[NAV] *** EMERGENCY STOP *** current and future actions halted")
                    stop_issued = True
                    _was_paused = True
                    _was_estopped = True
                    break

            if stop_issued:
                continue

            if not use_camera:
                break

            # Record when this batch of actions finished so the next
            # camera read is forced to wait for a newer post-action frame.
            next_frame_ref = cam.snapshot() if hasattr(cam, "snapshot") else None

    except KeyboardInterrupt:
        print("\n[NAV] Interrupted.")
        robot.stop()
    finally:
        if hasattr(robot, 'close'):
            robot.close()
        if cam:
            cam.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if vllm_proc and vllm_proc.poll() is None:
            print("[vLLM] Shutting down server…")
            os.killpg(os.getpgid(vllm_proc.pid), signal.SIGTERM)
            vllm_proc.wait(timeout=10)


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ActiveVLN checkpoint server → Unitree Go1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--instruction", default="",
                   help="Navigation instruction (leave empty to start paused, waiting for Web UI)")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible API base URL (if already running; skip --use-vllm)")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--action-space", choices=["r2r", "rxr"], default="r2r",
                   help="r2r: ±15/30/45° turns  rxr: ±30/60/90° turns")
    p.add_argument("--max-turns",  type=int, default=40,
                   help="Max LLM inference calls per episode")
    p.add_argument("--max-steps",  type=int, default=120,
                   help="Max primitive actions per episode")
    p.add_argument("--camera-id",  type=int, default=0)
    p.add_argument("--rs", action="store_true",
                   help="Use Intel RealSense color stream for RGB input instead of the default webcam")
    p.add_argument("--rs-width", type=int, default=640,
                   help="RealSense color stream width")
    p.add_argument("--rs-height", type=int, default=480,
                   help="RealSense color stream height")
    p.add_argument("--rs-fps", type=int, default=30,
                   help="RealSense color stream FPS")
    p.add_argument("--settle-time", type=float, default=0.4,
                   help="Seconds to wait after actions complete before capturing next frame")
    p.add_argument("--frame-timeout", type=float, default=4.0,
                   help="Seconds to wait for a fresh post-action camera frame before restarting the stream")
    p.add_argument("--image-path", default=None,
                   help="Single saved image (skips camera, one inference step)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print robot commands instead of executing them")
    p.add_argument("--headless", action="store_true",
                   help="Disable OpenCV display window")
    p.add_argument("--test-mode", action="store_true",
                   help="Clamp executed actions to FORWARD 25cm and TURN values above 15deg to 30deg")
    # vLLM server auto-launch (matches infer.py/infer.sh convention)
    p.add_argument("--use-vllm", action="store_true",
                   help="Auto-launch the vLLM checkpoint server before navigating")
    p.add_argument("--checkpoint-path",
                   default="checkpoints/Qwen2.5-VL-3B_rl_r2r_4000",
                   help="HuggingFace checkpoint path (used with --use-vllm)")
    p.add_argument("--vllm-port", type=int, default=8003)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.60)
    p.add_argument("--vllm-max-model-len", type=int, default=32768)
    p.add_argument("--vllm-log", default="deploy/vllm_server.log")
    # ── New feature flags ────────────────────────────────────────────────────
    p.add_argument("--enable-obstacle-replanning", action="store_true",
                   help="After each action, check RealSense depth and trigger VLM "
                        "re-inference with a replanning prefix when an obstacle is "
                        "detected. Requires --rs.")
    p.add_argument("--obstacle-depth-threshold", type=float, default=0.8,
                   help="Depth threshold in metres for obstacle detection (default 0.8)")
    p.add_argument("--enable-memory", action="store_true",
                   help="Build a running dead-reckoning spatial map across steps and "
                        "inject a text summary into every VLM prompt.")
    p.add_argument("--enable-landmark-tracking", action="store_true",
                   help="Extract landmark phrases from the instruction and log proximity "
                        "checkpoints as the agent progresses. Included in memory summary.")
    # Diadem AGV direct USB control
    p.add_argument("--diadem", action="store_true",
                   help="Use Diadem AGV over local USB micro-ROS instead of Unitree Go1")
    p.add_argument("--diadem-dev", default="/dev/esp",
                   help="Diadem ESP32 serial device")
    p.add_argument("--diadem-agent-wait", type=float, default=8.0,
                   help="Seconds to wait for ESP32 after starting micro_ros_agent")
    p.add_argument("--diadem-baud", type=int, default=921600,
                   help="Diadem ESP32 serial baud rate")
    p.add_argument("--diadem-ros-domain", type=int, default=101,
                   help="ROS_DOMAIN_ID used by the Diadem firmware")
    p.add_argument("--diadem-agent-log", default="/tmp/agv_micro_ros_agent.log",
                   help="Path for micro_ros_agent stdout/stderr")
    return p


if __name__ == "__main__":
    navigate(build_parser().parse_args())
