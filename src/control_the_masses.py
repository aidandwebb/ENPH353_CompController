#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Wrench, Twist
from sensor_msgs.msg import Imu
import tf.transformations as tft
import numpy as np
from std_msgs.msg import String
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


## tottally from chat idk if this will work?
class OpticalFlow:
    def __init__(self):
        self.bridge = CvBridge()
        self.prev = None

        rospy.Subscriber("/D1/down_cam/image_raw", Image, self.cb)

        self.vel = np.zeros(2)

    def cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.prev is None:
            self.prev = gray
            return

        flow = cv2.calcOpticalFlowFarneback(
            self.prev, gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Average flow
        fx = np.mean(flow[..., 0])
        fy = np.mean(flow[..., 1])

        self.vel = np.array([fx, fy])


        self.prev = gray


class DroneController:
    def __init__(self):
        rospy.init_node('drone_controller')

        self.pub = rospy.Publisher("/D1/force_cmd", Wrench, queue_size=10)
        rospy.Subscriber("/D1/imu", Imu, self.imu_cb)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_cb)

        self.roll = 0.0
        self.pitch = 0.0
        self.ang_vel = (0, 0, 0)
        self.cmd = Twist()
        self.R = np.eye(3)
        self.altitude = 0.0
        self.desired_altitude = 8.0
        self.mass = 15.18
        self.g = 9.81

        ## ---------- TUNING CONSTANTS ------------##
        self.k_flow = 0.8
        self.k_z = 30.0
        self.deadband = 0.01
    
        ## ----------------------------------------##

        ## I want to look at camera and be able to tell drifting and stuff from the pixels
        self.flow = OpticalFlow()

        self.rate = rospy.Rate(50)

    def imu_cb(self, msg):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        self.R = tft.quaternion_matrix(quat)[:3, :3]
        self.roll, self.pitch, _ = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.ang_vel = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
        

    def cmd_cb(self, msg):
        self.cmd = msg

    def run(self):

        while not rospy.is_shutdown():
            w = Wrench()

            # get drifts from camera maybe 
            drift_x = -self.k_flow * self.flow.vel[0]
            drift_y = -self.k_flow * self.flow.vel[1]

            if abs(drift_x) < self.deadband:
                drift_x = 0.0
            if abs(drift_y) < self.deadband:
                drift_y = 0.0

            print(f"x drift: {drift_x}, y drift: {drift_y}")

            torque_body = [
                -1*self.roll - 0.2*self.ang_vel[0],
                -1*self.pitch - 0.2*self.ang_vel[1],
                1 * self.cmd.angular.z - 0.25*self.ang_vel[2]
            ]

            # Rotate torque from body frame -> world frame using IMU quaternion
            torque_world = self.R.dot(torque_body)

            w.torque.x = torque_world[0]
            w.torque.y = torque_world[1]
            w.torque.z = torque_world[2]

            # find out how off from desired altitude we are and adjust
            # z_error = np.clip(self.desired_altitude - self.altitude, -2.0, 2.0)
            
            force_body_z = (100.0 * self.cmd.linear.z) + (self.mass * self.g) #+ (self.k_z * z_error) 

            force_body = [
                100.0 * (self.cmd.linear.x), #+ drift_x, 
                100.0 * (self.cmd.linear.y), #+ drift_y,  
                0
            ]

            # Rotate forces to world frame
            force_world = self.R.dot(force_body)

            w.force.x = force_world[0]
            w.force.y = force_world[1]
            w.force.z = force_body_z

            # Debug log
            error_log = f"command: {self.cmd.linear}, force_body = {force_body}, force_world = {force_world}"
            rospy.loginfo_throttle(1, error_log)

            self.pub.publish(w)
            self.rate.sleep()


if __name__ == "__main__":
    DroneController().run()