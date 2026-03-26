#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

from std_msgs.msg import String

class MissionStateMachine:
    def __init__(self):
        rospy.init_node("mission_state_machine")

        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size=1)


        self.state = "TAKEOFF"
        self.state_start = rospy.Time.now()

        self.takeoff_time = 50.0
        self.move_time = 5.0

        self.rate = rospy.Rate(20)

    def publish_for_duration(self, cmd, duration):
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < duration and not rospy.is_shutdown():
            self.pub.publish(cmd)
            self.rate.sleep()


    def run(self):
        rospy.sleep(1)

        # Start signal
        self.score_tracker.publish("test,1234,0,ABCD")
        rospy.sleep(1)

        # -------- TAKEOFF --------
        rospy.loginfo("Taking off")
        cmd = Twist()
        cmd.linear.z = 1.0
        self.publish_for_duration(cmd, 0.25)

        # -------- HOVER --------
        rospy.loginfo("Hover")
        cmd = Twist()
        self.publish_for_duration(cmd, 1.0)

        # -------- MOVE --------
        rospy.loginfo("Moving")
        cmd = Twist()
        cmd.linear.x = 1.0
        cmd.linear.y = -1.0
        self.publish_for_duration(cmd, 0.5)

        # -------- STOP --------
        rospy.loginfo("Stopping")
        cmd = Twist()
        self.publish_for_duration(cmd, 1.0)

        # -------- DONE --------
        rospy.loginfo("Done")
        self.score_tracker.publish("test,1234,-1,ABCD")


if __name__ == "__main__":
    MissionStateMachine().run()