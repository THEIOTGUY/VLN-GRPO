
import re
from typing import List, Tuple

# Copying the logic from deploy/go1_nav.py
_THINK_RE  = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_FWD_RE    = re.compile(r'\bmove forward (25|50|75)cm\b',     re.IGNORECASE)
_LEFT_R2R  = re.compile(r'\bturn left (15|30|45) degrees\b',  re.IGNORECASE)
_RIGHT_R2R = re.compile(r'\bturn right (15|30|45) degrees\b', re.IGNORECASE)
_LEFT_RXR  = re.compile(r'\bturn left (30|60|90) degrees\b',  re.IGNORECASE)
_RIGHT_RXR = re.compile(r'\bturn right (30|60|90) degrees\b', re.IGNORECASE)

def parse_action_text_original(
    raw: str,
    action_space: str = "r2r",
    max_actions: int = 3,
) -> List[Tuple[int, float]]:
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

    return out

test_cases = [
    "move forward 25cm, turn left 15 degrees, move forward 25cm",
    "move forward 25cm. turn left 15 degrees. move forward 25cm", # Uses dots
    "move forward 25cm turn left 15 degrees move forward 25cm", # No separator
    "1. move forward 25cm, 2. turn left 15 degrees, 3. move forward 50cm",
    "<think>I should move</think> move forward 25cm, turn left 15 degrees",
    "move forward 25 cm, turn left 15 degree", # Spaces and singular
]

for tc in test_cases:
    print(f"Input: {tc!r}")
    try:
        res = parse_action_text_original(tc)
        print(f"Result: {res}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 20)
