#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

class TeleopDrone:
    def __init__(self):
        rospy.init_node('teleop_drone')

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.settings = termios.tcgetattr(sys.stdin)

        self.cmd = Twist()
        self.rate = rospy.Rate(20)

        print("""
Controls:
  w/s : forward/back
  a/d : left/right
  r/f : up/down
  q/e : yaw
  x   : stop
""")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        target_height = 0
        try:
            while not rospy.is_shutdown():
                key = self.get_key()

                # Reset each loop (no accumulation)
                self.cmd = Twist()

                if key == 'w': self.cmd.linear.x = 1.0
                elif key == 's': self.cmd.linear.x = -1.0
                elif key == 'a': self.cmd.linear.y = 1.0
                elif key == 'd': self.cmd.linear.y = -1.0
                elif key == 'r': target_height += 0.1
                elif key == 'f': target_height -= 0.1
                elif key == 'q': self.cmd.angular.z = 1.0
                elif key == 'e': self.cmd.angular.z = -1.0
                elif key == 'x': self.cmd = Twist()
                elif key == '\x03': break
                self.cmd.linear.z = target_height
                self.pub.publish(self.cmd)
                self.rate.sleep()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

if __name__ == "__main__":
    TeleopDrone().run()