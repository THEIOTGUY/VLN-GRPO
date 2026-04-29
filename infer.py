#!/usr/bin/env python3
"""
ActiveVLN inference for Unitree Go1 on NVIDIA AGX Orin 32GB.

Model : Qwen2.5-VL-3B  (ActiveVLN RL checkpoint, global_step_450)
Robot : Unitree Go1 (unitree_legged_sdk HighCmd interface)
Camera: OpenCV webcam

Usage
-----
python infer.py --instruction "Turn left at the corridor and stop at the chair" \
                --checkpoint-path ./global_step_450/actor/huggingface
"""

import math
import os
import random
import re
import sys
import time
import base64
import subprocess
import signal
import urllib.request
from io import BytesIO

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import argparse
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def encode_pil_image_to_data_url(pil_image: Image.Image) -> str:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    max_ratio = 200
    max_pixels = max_pixels if max_pixels is not None else 16384 * factor ** 2
    min_pixels = min_pixels if min_pixels is not None else 4 * factor ** 2
    assert max_pixels >= min_pixels, "max_pixels must be >= min_pixels"
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


# ─────────────────────────────────────────────────────────────────
# Navigation prompts  (verbatim from eval/vlnce/eval_vlnce.py)
# ─────────────────────────────────────────────────────────────────

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

# Text appended after the image — matches vlnce_server/env.py obs_str construction:
# init_observation_template(...) + "\n" + format_prompt_text
# where format_prompt_text = "You can take up to 3 actions at a time, separated by ','. "
USER_PROMPT_SUFFIX = (
    "\nInstruction: {instruction}\n"
    "Decide your next action. \n"
    "You can take up to 3 actions at a time, separated by ','. "
)

# ─────────────────────────────────────────────────────────────────
# Go1 robot controller  (mirrors example_keyboard.cpp UDP approach)
# ─────────────────────────────────────────────────────────────────

class Go1Controller:
    """
    Unitree Go1 HighCmd controller using the same UDP interface as
    example_keyboard.cpp:  sdk.UDP(HIGHLEVEL, 8090, ROBOT_IP, 8082).

    Motion is timed: send velocity commands for (distance/speed) seconds
    then send an idle stop command — identical to the C++ discrete logic.
    Set dry_run=True to print commands without moving.
    """

    HIGHLEVEL  = 0xee
    ROBOT_IP   = "192.168.123.161"
    LOCAL_PORT = 8090
    ROBOT_PORT = 8082

    VX_FWD   =  0.4   # m/s  forward
    VX_BACK  =  0.3   # m/s  backward (magnitude)
    WZ_TURN  =  0.5   # rad/s turn

    CONTROL_DT = 0.002  # 500 Hz — same as C++ control loop

    SDK_LIB_PATH = "/home/chitti/VLN-DAPO-main/unitree/unitree_legged_sdk/lib/python/arm64"

    def __init__(self, dry_run: bool = False):
        self.dry_run  = dry_run
        self._udp     = None
        self._cmd     = None
        self._state   = None

        if not dry_run:
            try:
                sys.path.insert(0, self.SDK_LIB_PATH)
                import robot_interface as sdk
                self._sdk   = sdk
                self._udp   = sdk.UDP(self.HIGHLEVEL, self.LOCAL_PORT,
                                      self.ROBOT_IP,  self.ROBOT_PORT)
                self._cmd   = sdk.HighCmd()
                self._state = sdk.HighState()
                self._udp.InitCmdData(self._cmd)
                print("[Go1] SDK connected via UDP.")
            except Exception as exc:
                print(f"[Go1] SDK unavailable ({exc}) — dry-run mode.")
                self.dry_run = True

    def _make_cmd(self, vx: float, wz: float, moving: bool):
        cmd = self._cmd
        cmd.mode        = 2 if moving else 0
        cmd.gaitType    = 1 if moving else 0
        cmd.velocity    = [vx, 0.0]
        cmd.yawSpeed    = wz
        cmd.footRaiseHeight = 0.1 if moving else 0.0
        cmd.bodyHeight  = 0.0
        cmd.euler       = [0.0, 0.0, 0.0]
        cmd.speedLevel  = 0
        cmd.reserve     = 0

    def _send_timed(self, vx: float, wz: float, duration: float):
        """Send (vx, wz) for `duration` seconds at 500 Hz, then stop."""
        if self.dry_run or self._udp is None:
            print(f"  [DRY-RUN] vx={vx:+.2f}  wz={wz:+.2f}  t={duration:.2f}s")
            time.sleep(duration)
            return

        deadline = time.time() + duration
        while time.time() < deadline:
            self._udp.Recv()
            self._udp.GetRecv(self._state)
            self._make_cmd(vx, wz, moving=True)
            self._udp.SetSend(self._cmd)
            self._udp.Send()
            time.sleep(self.CONTROL_DT)

        self._make_cmd(0.0, 0.0, moving=False)
        self._udp.SetSend(self._cmd)
        self._udp.Send()

    def move_forward(self, distance_cm: float):
        duration = (distance_cm / 100.0) / self.VX_FWD
        print(f"[Go1] Forward {distance_cm:.0f} cm  ({duration:.2f} s)")
        self._send_timed(self.VX_FWD, 0.0, duration)

    def move_backward(self, distance_cm: float):
        duration = (distance_cm / 100.0) / self.VX_BACK
        print(f"[Go1] Backward {distance_cm:.0f} cm  ({duration:.2f} s)")
        self._send_timed(-self.VX_BACK, 0.0, duration)

    def turn_left(self, angle_deg: float):
        duration = math.radians(angle_deg) / self.WZ_TURN
        print(f"[Go1] Turn left  {angle_deg:.0f}°  ({duration:.2f} s)")
        self._send_timed(0.0, +self.WZ_TURN, duration)

    def turn_right(self, angle_deg: float):
        duration = math.radians(angle_deg) / self.WZ_TURN
        print(f"[Go1] Turn right {angle_deg:.0f}°  ({duration:.2f} s)")
        self._send_timed(0.0, -self.WZ_TURN, duration)

    def stop(self):
        print("[Go1] STOP")
        if not self.dry_run and self._udp is not None:
            self._make_cmd(0.0, 0.0, moving=False)
            self._udp.SetSend(self._cmd)
            self._udp.Send()
            time.sleep(0.1)
        else:
            print("  [DRY-RUN] stop")


# ─────────────────────────────────────────────────────────────────
# VLN agent (vLLM / OpenAI-compatible API)
# ─────────────────────────────────────────────────────────────────

class ActiveVLNAgent:
    """
    Queries Qwen2.5-VL-3B via an OpenAI-compatible API (vLLM) for
    multi-turn VLN inference.  Mirrors the logic in
    eval/vlnce/eval_vlnce.py.
    """

    MIN_PIXELS   = 1_024    # matches training agent.min_pixels
    MAX_PIXELS   = 76_800   # matches training/eval agent.max_pixels
    PIXEL_FACTOR = 28       # Qwen patch-size alignment

    def __init__(
        self,
        action_space     : str          = "r2r",
        max_turns        : int          = 40,
        max_steps        : int          = 120,
        base_url         : Optional[str] = None,
        api_key          : str          = "EMPTY",
        served_model_name: Optional[str] = None,
    ):
        self.action_space = action_space
        self.max_turns    = max_turns
        self.max_steps    = max_steps
        self.forward_step = 25
        self.turn_step    = 15 if action_space == "r2r" else 30
        self._step_count  = 0

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        if served_model_name:
            self.model_name = served_model_name
        else:
            self.model_name = self.client.models.list().data[0].id
        print(f"[VLN] Using OpenAI-compatible API backend at {base_url} (model={self.model_name})")
        self._reset_state()
        print("[VLN] Ready.\n")

    # ── conversation state ──────────────────────────────────────────

    def _reset_state(self):
        system_prompt = SYSTEM_PROMPT_R2R if self.action_space == "r2r" else SYSTEM_PROMPT_RXR
        self._conv: List[dict] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        self._pending    : List[Tuple[int, float]] = []
        self._turn_count = 0
        self._step_count = 0
        self._done       = False

    def reset(self):
        """Call between navigation episodes."""
        self._reset_state()

    # ── image helper ───────────────────────────────────────────────

    def _preprocess_image(self, bgr: np.ndarray) -> Image.Image:
        """BGR uint8 (OpenCV) → resized PIL RGB."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        rh, rw = smart_resize(
            pil.height, pil.width,
            max_pixels = self.MAX_PIXELS,
            factor     = self.PIXEL_FACTOR,
        )
        return pil.resize((rw, rh), Image.LANCZOS)

    # ── model call ─────────────────────────────────────────────────

    def _generate(self) -> str:
        outputs = self.client.chat.completions.create(
            messages=self._conv,
            model=self.model_name,
            max_completion_tokens=512,
            temperature=0.2,
            top_p=0.95,
        )
        return outputs.choices[0].message.content.strip()

    # ── action parsing ─────────────────────────────────────────────

    def _parse_raw(self, raw: str) -> List[Tuple[Optional[int], Optional[float]]]:
        """Parse comma-separated actions like 'move forward 50cm, turn left 15 degrees'."""
        return [self._parse_one(p.strip()) for p in raw.split(",")]

    def _parse_one(self, text: str) -> Tuple[Optional[int], Optional[float]]:
        t = text.lower()
        if "stop" in t:
            return 0, None
        if "forward" in t:
            m = re.search(r"-?\d+", t)
            return 1, float(m.group()) if m else float(self.forward_step)
        if "left" in t:
            m = re.search(r"-?\d+", t)
            return 2, float(m.group()) if m else float(self.turn_step)
        if "right" in t:
            m = re.search(r"-?\d+", t)
            return 3, float(m.group()) if m else float(self.turn_step)
        return None, None

    def _expand(self, parsed: List[Tuple[Optional[int], Optional[float]]]) -> List[Tuple[int, float]]:
        """Expand multi-cm / multi-degree actions to primitive list."""
        out: List[Tuple[int, float]] = []
        for aid, num in parsed:
            if aid == 0:
                out.append((0, 0.0))
            elif aid == 1 and num is not None:
                for _ in range(min(3, max(1, int(num / self.forward_step)))):
                    out.append((1, float(self.forward_step)))
            elif aid == 2 and num is not None:
                for _ in range(min(3, max(1, int(num / self.turn_step)))):
                    out.append((2, float(self.turn_step)))
            elif aid == 3 and num is not None:
                for _ in range(min(3, max(1, int(num / self.turn_step)))):
                    out.append((3, float(self.turn_step)))
        if not out:
            fb = random.randint(1, 3)
            out.append((fb, float(self.forward_step if fb == 1 else self.turn_step)))
        return out

    # ── public step API ────────────────────────────────────────────

    def step(
        self,
        bgr_frame  : np.ndarray,
        instruction: str,
    ) -> Tuple[int, float, str]:
        """
        Given a BGR camera frame and instruction, return the next action.

        Returns
        -------
        action_id  : 0=stop  1=forward  2=turn_left  3=turn_right
        numeric    : cm (forward) or degrees (turn); 0 for stop
        raw_text   : model output string (empty when draining pending queue)
        """
        if self._done:
            return 0, 0.0, "done"

        if self._pending:
            if self._step_count >= self.max_steps:
                print(f"[VLN] Max steps ({self.max_steps}) reached — stopping.")
                self._done = True
                return 0, 0.0, "max_steps"
            aid, num = self._pending.pop(0)
            self._step_count += 1
            return aid, num, ""

        self._turn_count += 1
        if self._turn_count > self.max_turns:
            print(f"[VLN] Max turns ({self.max_turns}) reached — stopping.")
            self._done = True
            return 0, 0.0, "max_turns"

        pil    = self._preprocess_image(bgr_frame)

        # Write the exact preprocessed frame to the shared file so the UI
        # shows what the model actually sees (same crop, same resolution).
        try:
            preview_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            cv2.imwrite(Camera.SHARED_FRAME, preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception:
            pass

        # Prefix includes trailing \n to match training obs_str:
        # "[Initial Observation]:\n<image>\nInstruction:..." (vlnce_server/prompt.py)
        prefix = "[Initial Observation]:\n" if self._turn_count == 1 else "After that, the observation is:\n"
        self._conv.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prefix},
                {"type": "image_url", "image_url": {"url": encode_pil_image_to_data_url(pil)}},
                {"type": "text", "text": USER_PROMPT_SUFFIX.format(instruction=instruction)},
            ],
        })

        raw = self._generate()
        self._conv.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})

        primitives = self._expand(self._parse_raw(raw))
        if self._step_count >= self.max_steps:
            print(f"[VLN] Max steps ({self.max_steps}) reached — stopping.")
            self._done = True
            return 0, 0.0, "max_steps"
        first = primitives.pop(0)
        self._step_count += 1
        self._pending.extend(primitives)

        if first[0] == 0:
            self._done = True

        return first[0], first[1], raw


# ─────────────────────────────────────────────────────────────────
# Thread-safe camera capture
# ─────────────────────────────────────────────────────────────────

class Camera:
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480,
                 shared_frame_fps: float = 4.0):
        self._cap = cv2.VideoCapture(camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        self._frame: Optional[np.ndarray] = None
        self._lock  = threading.Lock()
        self._shared_frame_interval = 0.0 if shared_frame_fps <= 0 else 1.0 / shared_frame_fps
        self._last_shared_frame_at = 0.0
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        time.sleep(0.6)

    SHARED_FRAME = "/tmp/vln_latest_frame.jpg"

    def _loop(self):
        while True:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                now = time.time()
                if now - self._last_shared_frame_at >= self._shared_frame_interval:
                    try:
                        cv2.imwrite(self.SHARED_FRAME, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        self._last_shared_frame_at = now
                    except Exception:
                        pass

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._cap.release()


# ─────────────────────────────────────────────────────────────────
# vLLM server helpers
# ─────────────────────────────────────────────────────────────────

def _start_vllm_server(
    model_path: str,
    port: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    log_path: str,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--dtype", dtype,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--allowed-origins", '["*"]',
        "--disable-log-requests",
        # Match training config (train_vlnce_4gpus.yaml): chunked prefill off.
        "--no-enable-chunked-prefill",
        # Match training: max_num_batched_tokens: 32768.
        "--max-num-batched-tokens", "32768",
        # Disable torch.compile: torch inductor level-3 fails on Jetson
        # (torch 2.8.0 + inductor pattern-matcher AssertionError).
        "--enforce-eager",
        # Match training: max_vllm_images=200, max_vllm_videos=0.
        # video=0 disables video profiling which OOMs on Orin unified memory.
        "--limit-mm-per-prompt", '{"image": 200, "video": 0}',
        # Cap image resolution to match training (max_pixels: 76800 in agent
        # config). Limits dummy image visual tokens during profiling forward pass.
        "--mm-processor-kwargs", '{"max_pixels": 76800, "min_pixels": 1024}',
    ]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Expose Jetson system CUDA headers so vllm can JIT-compile CUDA extensions.
    _cuda_home = "/usr/local/cuda-12.6"
    _cuda_inc = f"{_cuda_home}/targets/aarch64-linux/include"
    env["CUDA_HOME"] = _cuda_home
    env["CUDA_PATH"] = _cuda_home
    env["CPATH"] = f"{_cuda_inc}:{env.get('CPATH', '')}"
    env["PATH"] = f"{_cuda_home}/bin:{env.get('PATH', '')}"
    # Triton resolves ptxas via knobs (not PATH); set explicitly.
    env.setdefault("TRITON_PTXAS_PATH", f"{_cuda_home}/bin/ptxas")
    log_f = open(log_path, "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=log_f,
        start_new_session=True,
        env=env,
    )
    print(f"[vLLM] Server PID={proc.pid}  model={model_path}  port={port}  log={log_path}")
    return proc


def _wait_for_vllm(port: int, timeout: int = 120) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    print(f"[vLLM] Waiting for server on port {port} (up to {timeout}s)…")
    for i in range(timeout):
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    print(f"[vLLM] Server ready after {i+1}s.")
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ─────────────────────────────────────────────────────────────────
# Main navigation loop
# ─────────────────────────────────────────────────────────────────

def navigate(args):
    vllm_proc: Optional[subprocess.Popen] = None

    if args.use_vllm:
        if args.base_url:
            print(f"[vLLM] --base-url already set ({args.base_url}); skipping auto-launch.")
        else:
            dtype_str = "auto"  # matches training (auto → bfloat16 on Orin SM8.7)
            vllm_proc = _start_vllm_server(
                model_path             = args.checkpoint_path,
                port                   = args.vllm_port,
                gpu_memory_utilization = args.vllm_gpu_memory_utilization,
                max_model_len          = args.vllm_max_model_len,
                dtype                  = dtype_str,
                log_path               = args.vllm_log,
            )
            if not _wait_for_vllm(args.vllm_port, timeout=360):
                vllm_proc.terminate()
                raise RuntimeError(f"vLLM server did not become ready on port {args.vllm_port}")
            args.base_url = f"http://127.0.0.1:{args.vllm_port}/v1"
            print(f"[vLLM] Using API backend at {args.base_url}")

    agent = ActiveVLNAgent(
        action_space      = args.action_space,
        max_turns         = args.max_turns,
        max_steps         = args.max_steps,
        base_url          = args.base_url,
        api_key           = args.api_key,
        served_model_name = args.served_model_name,
    )
    robot = Go1Controller(dry_run=args.dry_run)

    use_camera = args.image_path is None
    cam: Optional[Camera] = None
    if use_camera:
        cam = Camera(camera_id=args.camera_id, shared_frame_fps=args.shared_frame_fps)

    print(f"\n{'='*60}")
    print(f"  Instruction : {args.instruction}")
    print(f"  Action space: {args.action_space}")
    print(f"  Max turns   : {args.max_turns}  (LLM calls)")
    print(f"  Max steps   : {args.max_steps}  (primitive actions)")
    print(f"  Dry-run     : {args.dry_run}")
    print(f"{'='*60}\n")

    step = 0
    _instr_file = "/tmp/vln_instruction.txt"
    _pause_flag = "/tmp/vln_paused.flag"
    _last_instruction = args.instruction
    _was_paused = False
    try:
        while True:
            # ── pause gate ──────────────────────────────────────
            if os.path.exists(_pause_flag):
                if not _was_paused:
                    print("[VLN] *** PAUSED *** waiting for new instruction…")
                    _was_paused = True
                time.sleep(0.5)
                continue
            elif _was_paused:
                print("[VLN] *** RESUMED ***")
                _was_paused = False

            # ── grab frame ──────────────────────────────────────
            if use_camera:
                frame = cam.read()
                if frame is None:
                    time.sleep(0.05)
                    continue
            else:
                frame = cv2.imread(args.image_path)
                if frame is None:
                    raise FileNotFoundError(args.image_path)

            # ── optional live preview ────────────────────────────
            if not args.headless:
                cv2.imshow("ActiveVLN — Go1", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[NAV] User quit.")
                    break

            # ── check for restart signal ─────────────────────────
            _restart_flag = "/tmp/vln_restart.flag"
            if os.path.exists(_restart_flag):
                try:
                    _new_instr = open(_restart_flag).read().strip()
                    os.remove(_restart_flag)
                    agent._conv.clear()
                    agent._pending.clear()
                    agent._done = False
                    agent._turn_count = 0
                    agent._step_count = 0
                    step = 0
                    if _new_instr:
                        args.instruction = _new_instr
                        _last_instruction = _new_instr
                        open(_instr_file, 'w').write(_new_instr)
                        print(f"\n[VLN] *** RESTART *** instruction={args.instruction}\n")
                    else:
                        open(_pause_flag, 'w').write("1")
                        print("\n[VLN] *** RESTART *** pausing for new instruction\n")
                        continue
                except Exception:
                    pass

            # ── check for live instruction change ────────────────
            if os.path.exists(_instr_file):
                try:
                    _live = open(_instr_file).read().strip()
                    if _live and _live != _last_instruction:
                        args.instruction = _live
                        _last_instruction = _live
                        agent._conv.clear()
                        agent._pending.clear()
                        agent._done = False
                        agent._turn_count = 0
                        agent._step_count = 0
                        step = 0
                        if os.path.exists(_pause_flag):
                            os.remove(_pause_flag)
                        print(f"[VLN] *** NEW INSTRUCTION *** → {_live}")
                except Exception:
                    pass

            # ── inference ───────────────────────────────────────
            step += 1
            t0 = time.time()
            action_id, numeric, raw = agent.step(frame, args.instruction)
            elapsed = time.time() - t0

            label = {0: "STOP", 1: f"FORWARD {numeric:.0f}cm",
                     2: f"TURN LEFT {numeric:.0f}°",
                     3: f"TURN RIGHT {numeric:.0f}°"}.get(action_id, "?")
            print(f"[Step {step:03d}] {elapsed:.1f}s  →  {label}"
                  + (f"   ('{raw}')" if raw else ""))

            # ── execute ─────────────────────────────────────────
            if action_id == 0:
                robot.stop()
                stop_reason = {
                    "max_steps": "max steps reached",
                    "max_turns": "max turns reached",
                    "done":      "already stopped",
                }.get(raw, "STOP command")
                print(f"\n[NAV] Navigation complete ({stop_reason}). Waiting for next instruction…")
                agent.reset()
                step = 0
                _last_instruction = ""  # allow re-sending same instruction
                _was_paused = True      # suppress duplicate print in pause gate
                with open(_pause_flag, 'w') as _pf:
                    _pf.write("1")
                print("[VLN] *** PAUSED *** waiting for new instruction…")
                continue
            elif action_id == 1:
                robot.move_forward(numeric)
            elif action_id == 2:
                robot.turn_left(numeric)
            elif action_id == 3:
                robot.turn_right(numeric)

            if not use_camera:
                break

    except KeyboardInterrupt:
        print("\n[NAV] Interrupted — stopping.")
        robot.stop()
    finally:
        if cam is not None:
            cam.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if vllm_proc is not None and vllm_proc.poll() is None:
            print("[vLLM] Shutting down vLLM server…")
            os.killpg(os.getpgid(vllm_proc.pid), signal.SIGTERM)
            vllm_proc.wait(timeout=10)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ActiveVLN — Unitree Go1 / AGX Orin inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-path",
        default="./global_step_450/actor/huggingface",
        help="HuggingFace-format model directory or HF hub ID",
    )
    p.add_argument("--base-url", default=None,
                   help="Use an existing OpenAI-compatible API backend instead of auto-launching vLLM")
    p.add_argument("--api-key", default="EMPTY",
                   help="API key for OpenAI-compatible backend")
    p.add_argument("--served-model-name", default=None,
                   help="Explicit model name for the OpenAI-compatible backend")
    p.add_argument(
        "--instruction", required=True,
        help="Natural-language navigation instruction",
    )
    p.add_argument(
        "--action-space", choices=["r2r", "rxr"], default="r2r",
        help="R2R: ±15/30/45° turns; RxR: ±30/60/90° turns",
    )
    p.add_argument("--max-turns", type=int, default=40,
                   help="Max LLM inference calls per episode")
    p.add_argument("--max-steps", type=int, default=120,
                   help="Max primitive actions per episode")
    p.add_argument("--shared-frame-fps", type=float, default=4.0,
                   help="Rate limit UI preview frame writes")
    p.add_argument(
        "--camera-id", type=int, default=0,
        help="OpenCV camera index for live navigation",
    )
    p.add_argument(
        "--image-path", default=None,
        help="Run one inference step on a saved image (no camera needed)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print robot commands instead of executing them",
    )
    p.add_argument(
        "--headless", action="store_true",
        help="Disable OpenCV display (for SSH / no-display sessions)",
    )
    p.add_argument(
        "--use-vllm", action="store_true",
        help="Auto-launch a vLLM server for the checkpoint and query it via OpenAI API",
    )
    p.add_argument("--vllm-port", type=int, default=8003,
                   help="Port for the auto-launched vLLM server")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.60,
                   help="gpu_memory_utilization for vLLM")
    p.add_argument("--vllm-max-model-len", type=int, default=8192,
                   help="max_model_len for vLLM server")
    p.add_argument("--vllm-log", default="/home/chitti/VLN-DAPO-main/vllm_server.log",
                   help="File to append vLLM server stdout/stderr to")
    return p


if __name__ == "__main__":
    navigate(build_parser().parse_args())
