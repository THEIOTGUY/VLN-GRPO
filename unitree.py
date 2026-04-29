#!/usr/bin/env python3
"""
Manual Unitree Go1 command runner for ActiveVLN.

Examples
--------
python3 unitree.py "turn right 45 degrees, turn right 15 degrees, turn right 15 degrees"
python3 unitree.py "move forward 75cm, move forward 25cm, turn right 15 degrees"
python3 unitree.py --dry-run
"""

import argparse
import os
import sys
import time
from typing import List, Tuple


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deploy.go1_nav import Go1Controller, parse_action_text


def _labels(actions: List[Tuple[int, float]]) -> List[str]:
    out = []
    for aid, val in actions:
        if aid == 0:
            out.append("STOP")
        elif aid == 1:
            out.append(f"FORWARD {val:.0f}cm")
        elif aid == 2:
            out.append(f"TURN LEFT {val:.0f}°")
        elif aid == 3:
            out.append(f"TURN RIGHT {val:.0f}°")
        else:
            out.append(f"UNKNOWN {aid} {val}")
    return out


def execute_actions(robot: Go1Controller, actions: List[Tuple[int, float]], raw: str):
    print(f"[unitree.py] raw='{raw}'")
    print(f"[unitree.py] parsed={_labels(actions)}")
    for idx, (action_id, numeric) in enumerate(actions, start=1):
        print(f"[unitree.py] action {idx}/{len(actions)}")
        if action_id == 0:
            robot.stop()
            print("[unitree.py] stop received, ending batch")
            break
        if action_id == 1:
            robot.move_forward(numeric)
        elif action_id == 2:
            robot.turn_left(numeric)
        elif action_id == 3:
            robot.turn_right(numeric)


def run_interactive(robot: Go1Controller, action_space: str):
    print("[unitree.py] interactive mode")
    print("[unitree.py] enter model-style actions, or 'quit' to exit")
    while True:
        try:
            raw = input("unitree> ").strip()
        except EOFError:
            print()
            break
        if not raw:
            continue
        if raw.lower() in {"quit", "exit"}:
            break
        actions = parse_action_text(raw, action_space=action_space, max_actions=3)
        execute_actions(robot, actions, raw)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manual ActiveVLN-compatible Unitree Go1 command runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "actions",
        nargs="*",
        help="Model-style action string, e.g. 'move forward 75cm, turn right 15 degrees'",
    )
    p.add_argument("--action-space", choices=["r2r", "rxr"], default="r2r")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without sending them to the robot")
    p.add_argument("--repeat", type=int, default=1,
                   help="Execute the same parsed batch this many times")
    p.add_argument("--pause", type=float, default=0.0,
                   help="Seconds to wait between repeated batches")
    return p


def main():
    args = build_parser().parse_args()
    raw = " ".join(args.actions).strip()
    robot = Go1Controller(dry_run=args.dry_run)
    try:
        if not raw:
            run_interactive(robot, args.action_space)
            return

        actions = parse_action_text(raw, action_space=args.action_space, max_actions=3)
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"[unitree.py] batch {i+1}/{args.repeat}")
            execute_actions(robot, actions, raw)
            if i + 1 < args.repeat and args.pause > 0:
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\n[unitree.py] interrupted")
        robot.stop()


if __name__ == "__main__":
    main()
