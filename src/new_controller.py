#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tf.transformations as tft

from geometry_msgs.msg import Wrench, Twist
from sensor_msgs.msg import Imu, Image
from cv_bridge import CvBridge


class DownwardMotionEstimator:
    def __init__(self):
        self.bridge = CvBridge()

        self.prev_gray = None
        self.prev_pts = None
        self.flow_px = np.zeros(2, dtype=np.float32)

        self.alpha = 0.25
        self.max_corners = 80
        self.quality = 0.01
        self.min_dist = 8
        self.reinit_period = 8
        self.frame_count = 0

        rospy.Subscriber("/D1/down_cam/image_raw", Image, self.image_cb, queue_size=1)

    def preprocess(self, gray):
        h, w = gray.shape
        roi = gray[h//4:3*h//4, w//4:3*w//4]
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        return roi

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self.preprocess(gray)

        self.frame_count += 1

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality,
                minDistance=self.min_dist,
                blockSize=7
            )
            return

        need_reinit = (
            self.prev_pts is None or
            len(self.prev_pts) < 15 or
            self.frame_count % self.reinit_period == 0
        )

        if need_reinit:
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality,
                minDistance=self.min_dist,
                blockSize=7
            )

        if self.prev_pts is None:
            self.prev_gray = gray
            return

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        if next_pts is None or status is None:
            self.prev_gray = gray
            self.prev_pts = None
            return

        good_old = self.prev_pts[status.flatten() == 1]
        good_new = next_pts[status.flatten() == 1]

        if len(good_old) >= 8:
            disp = good_new - good_old
            dx = np.median(disp[:, 0])
            dy = np.median(disp[:, 1])
            raw = np.array([dx, dy], dtype=np.float32)
            self.flow_px = self.alpha * raw + (1.0 - self.alpha) * self.flow_px

        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None


class DownwardAltitudeEstimator:
    def __init__(self):
        self.bridge = CvBridge()
        self.altitude = None
        self.alpha = 0.3

        rospy.Subscriber("/D1/down_cam/depth/image_raw", Image, self.depth_cb, queue_size=1)

    def depth_cb(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return

        if depth is None or depth.ndim != 2:
            return

        h, w = depth.shape
        patch = depth[h//2-12:h//2+12, w//2-12:w//2+12].astype(np.float32)

        valid = np.isfinite(patch) & (patch > 0.05) & (patch < 20.0)
        vals = patch[valid]

        if vals.size == 0:
            return

        z = float(np.median(vals))

        if self.altitude is None:
            self.altitude = z
        else:
            self.altitude = self.alpha * z + (1.0 - self.alpha) * self.altitude


class DroneController:
    def __init__(self):
        rospy.init_node("drone_controller")

        self.pub = rospy.Publisher("/D1/force_cmd", Wrench, queue_size=1)
        rospy.Subscriber("/D1/imu", Imu, self.imu_cb, queue_size=1)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_cb, queue_size=1)

        self.cmd = Twist()

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.ang_vel = np.zeros(3, dtype=np.float32)

        self.R = np.eye(3)
        self.mass = 15.17
        self.g = 9.81

        self.motion = DownwardMotionEstimator()
        self.alt = DownwardAltitudeEstimator()

        self.k_roll = 2.5
        self.k_pitch = 2.5
        self.k_p_rollrate = 0.35
        self.k_p_pitchrate = 0.35
        self.k_yaw = 1.2
        self.k_yawrate = 0.25

        self.k_flow = 0.20
        self.flow_deadband = 0.08

        self.kz_p = 5.0
        self.kz_d = 7.0
        self.prev_z_err = 0.0
        self.prev_t = rospy.Time.now()

        self.rate = rospy.Rate(60)

    def imu_cb(self, msg):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]

        self.R = tft.quaternion_matrix(quat)[:3, :3]
        self.roll, self.pitch, self.yaw = tft.euler_from_quaternion(quat)
        self.ang_vel[:] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    def cmd_cb(self, msg):
        self.cmd = msg

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = max((now - self.prev_t).to_sec(), 1e-3)
            self.prev_t = now

            w = Wrench()

            flow_x, flow_y = self.motion.flow_px
            if abs(flow_x) < self.flow_deadband:
                flow_x = 0.0
            if abs(flow_y) < self.flow_deadband:
                flow_y = 0.0

            drift_cmd_x = -self.k_flow * flow_x
            drift_cmd_y = -self.k_flow * flow_y

            body_force_xy = np.array([
                10.0 * self.cmd.linear.x + drift_cmd_x,
                10.0 * self.cmd.linear.y + drift_cmd_y,
                0.0
            ], dtype=np.float32)

            force_world_xy = self.R.dot(body_force_xy)

            desired_altitude = self.cmd.linear.z
            thrust_z = self.mass * self.g

            if self.alt.altitude is not None:
                z_err = desired_altitude - self.alt.altitude
                z_err_dot = (z_err - self.prev_z_err) / dt
                self.prev_z_err = z_err
                thrust_z += self.kz_p * z_err + self.kz_d * z_err_dot

            torque_body = np.array([
                -self.k_roll * self.roll - self.k_p_rollrate * self.ang_vel[0],
                -self.k_pitch * self.pitch - self.k_p_pitchrate * self.ang_vel[1],
                self.k_yaw * self.cmd.angular.z - self.k_yawrate * self.ang_vel[2]
            ], dtype=np.float32)

            torque_world = self.R.dot(torque_body)

            w.force.x = float(force_world_xy[0])
            w.force.y = float(force_world_xy[1])
            w.force.z = float(thrust_z)

            w.torque.x = float(torque_world[0])
            w.torque.y = float(torque_world[1])
            w.torque.z = float(torque_world[2])

            rospy.loginfo_throttle(
                0.2,
                f"alt={self.alt.altitude} target_z={desired_altitude:.2f} "
                f"flow=({self.motion.flow_px[0]:.2f},{self.motion.flow_px[1]:.2f}) "
                f"F=({w.force.x:.2f},{w.force.y:.2f},{w.force.z:.2f})"
            )

            self.pub.publish(w)
            self.rate.sleep()


if __name__ == "__main__":
    DroneController().run()