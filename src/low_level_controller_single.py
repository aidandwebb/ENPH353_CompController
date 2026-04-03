#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations as tft
import cv2

from geometry_msgs.msg import Wrench, Twist
from sensor_msgs.msg import Imu, Image

from cv_bridge import CvBridge


class DroneController:
    def __init__(self):
        rospy.init_node("drone_controller")

        self.pub = rospy.Publisher("/D1/force_cmd", Wrench, queue_size=10)

        rospy.Subscriber("/D1/imu", Imu, self.imu_cb, queue_size=1)
        rospy.Subscriber("/D1/down_cam/depth/image_raw", Image, self.depth_cb, queue_size=1)
        rospy.Subscriber("/D1/down_cam/image_raw", Image, self.image_cb)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_cb, queue_size=1)

        self.bridge = CvBridge()

        self.roll = 0.0
        self.pitch = 0.0
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)

        self.cmd = Twist()
        self.mass = 15.19
        self.g = 9.81

        self.altitude = None
        self.vertical_speed = 0.0
        self.last_altitude = None
        self.last_depth_time = None

        # Altitude PID constants:
        self.kz_p = 45.0
        self.kz_i = 6.0
        self.kz_d = 26.0

        # Road stuff
        self.x_error = 0
        self.road_angle = 0
        

        self.z_integral = 0.0
        self.z_integral_limit = 2.0
        self.max_z_correction = 120.0

        self.k_roll = 1.0
        self.k_pitch = 1.0
        self.k_rollrate = 0.2
        self.k_pitchrate = 0.2
        self.k_yaw = 1.0
        self.k_yawrate = 0.25

        self.k_xy = 10.0

        self.rate = rospy.Rate(50)

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # low saturation (gray)
        mask_s = cv2.inRange(s, 0, 5)

        # mid brightness (reject dark + very bright)
        mask_v = cv2.inRange(v, 120, 180)

        # combine
        mask = cv2.bitwise_and(mask_s, mask_v)

        mask = cv2.bitwise_not(mask)

        # --- find biggest blob ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        cnt = max(contours, key=cv2.contourArea)

        # --- get rotated rectangle ---
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        # --- fix angle (important) ---
        if w < h:
            angle = angle + 90  # make it consistent

        # --- lateral error ---
        img_center_x = img.shape[1] / 2
        x_error = cx - img_center_x

        # --- visualize ---
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        self.x_error = x_error
        self.road_angle = np.deg2rad(angle)

        cv2.imshow("rect", vis)
        cv2.waitKey(1)
            
    def imu_cb(self, msg):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        self.R = tft.quaternion_matrix(quat)[:3, :3].astype(np.float32)
        self.roll, self.pitch, _ = tft.euler_from_quaternion(quat)
        self.ang_vel[:] = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ]


    def depth_cb(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return

        if depth is None or depth.ndim != 2:
            return

        h, w = depth.shape
        patch = depth[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10].astype(np.float32)

        valid = np.isfinite(patch) & (patch > 0.05) & (patch < 20.0)
        vals = patch[valid]
        if vals.size == 0:
            return

        z = float(np.median(vals))
        now = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()

        if self.altitude is None:
            self.altitude = z
            self.last_altitude = z
            self.last_depth_time = now
            self.vertical_speed = 0.0
            return

        dt = max(now - self.last_depth_time, 1e-3)
        raw_vz = (z - self.last_altitude) / dt

        alpha_z = 0.35
        alpha_v = 0.25
        self.altitude = alpha_z * z + (1.0 - alpha_z) * self.altitude
        self.vertical_speed = alpha_v * raw_vz + (1.0 - alpha_v) * self.vertical_speed

        self.last_altitude = z
        self.last_depth_time = now

    def cmd_cb(self, msg):
        self.cmd = msg

    def run(self):
        last_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = max((now - last_time).to_sec(), 1e-3)
            last_time = now

            w = Wrench()

            torque_body = np.array([
                -self.k_roll * self.roll - self.k_rollrate * self.ang_vel[0],
                -self.k_pitch * self.pitch - self.k_pitchrate * self.ang_vel[1],
                self.k_yaw * self.cmd.angular.z - self.k_yawrate * self.ang_vel[2]
            ], dtype=np.float32)

            torque_world = self.R.dot(torque_body)

            w.torque.x = float(torque_world[0])
            w.torque.y = float(torque_world[1])
            w.torque.z = float(torque_world[2])

            force_body = np.array([
                self.k_xy * self.cmd.linear.x,
                self.k_xy * self.cmd.linear.y,
                0.0
            ], dtype=np.float32)

            force_world = self.R.dot(force_body)

            w.force.x = float(force_world[0])
            w.force.y = float(force_world[1])

            thrust_z = self.mass * self.g

            if self.altitude is not None:
                z_target = self.cmd.linear.z
                z_err = z_target - self.altitude

                self.z_integral += z_err * dt
                self.z_integral = float(np.clip(
                    self.z_integral,
                    -self.z_integral_limit,
                    self.z_integral_limit
                ))

                z_correction = (
                    self.kz_p * z_err +
                    self.kz_i * self.z_integral -
                    self.kz_d * self.vertical_speed
                )
                z_correction = float(np.clip(
                    z_correction,
                    -self.max_z_correction,
                    self.max_z_correction
                ))

                thrust_z += z_correction

            w.force.z = float(thrust_z)

            rospy.loginfo_throttle(
                0.2,
                f"alt={self.altitude} target={self.cmd.linear.z:.2f} vz={self.vertical_speed:.2f} "
                f"F=({w.force.x:.1f}, {w.force.y:.1f}, {w.force.z:.1f})"
            )

            self.pub.publish(w)
            self.rate.sleep()


if __name__ == "__main__":
    DroneController().run()