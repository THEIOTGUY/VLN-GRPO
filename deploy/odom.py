#!/usr/bin/env python3
"""
odom.py — Move the Diadem robot by centimetres and degrees.
Usage at prompt:
  50 90       → move forward 50 cm, then turn 90 degrees left
  50 -90      → move forward 50 cm, then turn 90 degrees right
  -30 0       → move backward 30 cm, no turn
  0 180       → turn 180 degrees in place
  stop        → stop immediately
  exit        → quit

Tune LINEAR_SPEED and ANGULAR_SPEED below if the distances/angles are off.
"""

import threading
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ── Tune these if actual distance/angle is off ─────────────────────────────
LINEAR_SPEED   = 0.3   # m/s forward/backward speed during move
ANGULAR_SPEED  = 0.6   # rad/s rotation speed during turn
# ───────────────────────────────────────────────────────────────────────────

class OdomNode(Node):
    def __init__(self):
        super().__init__('odom_controller')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._busy = False
        threading.Thread(target=self._input_loop, daemon=True).start()
        self.get_logger().info(
            "Ready. Enter: distance_cm  degrees  (e.g.  50 90)\n"
            "Commands: stop, exit"
        )

    def _publish(self, linear=0.0, angular=0.0):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.pub.publish(msg)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _move(self, distance_cm, degrees):
        self._busy = True
        try:
            # ── Forward / backward ─────────────────────────────────────────
            if abs(distance_cm) > 0.001:
                dist_m   = distance_cm / 100.0
                duration = abs(dist_m) / LINEAR_SPEED
                speed    = LINEAR_SPEED if dist_m > 0 else -LINEAR_SPEED
                print(f"  Moving {'forward' if dist_m > 0 else 'backward'} "
                      f"{abs(distance_cm):.1f} cm  ({duration:.2f} s)")
                t0 = time.time()
                while time.time() - t0 < duration:
                    self._publish(linear=speed)
                    time.sleep(0.05)
                self._stop()
                time.sleep(0.3)          # brief pause between move and turn

            # ── Turn ───────────────────────────────────────────────────────
            if abs(degrees) > 0.01:
                import math
                angle_rad = math.radians(abs(degrees))
                duration  = angle_rad / ANGULAR_SPEED
                ang_speed = ANGULAR_SPEED if degrees > 0 else -ANGULAR_SPEED
                print(f"  Turning {'left' if degrees > 0 else 'right'} "
                      f"{abs(degrees):.1f}°  ({duration:.2f} s)")
                t0 = time.time()
                while time.time() - t0 < duration:
                    self._publish(angular=ang_speed)
                    time.sleep(0.05)
                self._stop()

        finally:
            self._stop()
            self._busy = False
            print("  Done.\n")

    def _input_loop(self):
        while rclpy.ok():
            try:
                s = input("> ").strip().lower()
            except EOFError:
                break

            if s in ("exit", "quit"):
                self._stop()
                rclpy.shutdown()
                break

            if s in ("stop", "0", ""):
                self._stop()
                self._busy = False
                print("  Stopped.")
                continue

            if self._busy:
                print("  Still moving — type 'stop' to interrupt.")
                continue

            parts = s.split()
            if len(parts) != 2:
                print("  Bad input. Use: distance_cm  degrees  (e.g. 50 90)")
                continue

            try:
                distance_cm = float(parts[0])
                degrees     = float(parts[1])
            except ValueError:
                print("  Bad input. Numbers only.")
                continue

            threading.Thread(
                target=self._move, args=(distance_cm, degrees), daemon=True
            ).start()


def main():
    rclpy.init()
    node = OdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
