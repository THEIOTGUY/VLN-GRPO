#!/usr/bin/env python3
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelFromInput(Node):
    def __init__(self):
        super().__init__('cmd_vel_from_input')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.v = 0.0  # linear x (m/s)
        self.w = 0.0  # angular z (rad/s)

        # publish at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_cmd)

        # read terminal input without blocking ROS
        threading.Thread(target=self.input_loop, daemon=True).start()

        self.get_logger().info("Enter: v w   (example: 0.2 -0.3). Commands: stop, exit")

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x = self.v
        msg.angular.z = self.w
        self.pub.publish(msg)

    def input_loop(self):
        while rclpy.ok():
            s = input("> ").strip().lower()
            if s in ("exit", "quit"):
                self.v, self.w = 0.0, 0.0
                break
            if s in ("stop", "0"):
                self.v, self.w = 0.0, 0.0
                continue

            try:
                v_str, w_str = s.split()
                self.v = float(v_str)
                self.w = float(w_str)
            except Exception:
                print("Bad input. Use: v w (e.g., 0.2 0.4) or 'stop'/'exit'")

def main():
    rclpy.init()
    node = CmdVelFromInput()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
