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

        self.bridge = CvBridge()

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)

        self.mass = 15.19
        self.g = 9.81

        self.altitude = None
        self.vertical_speed = 0.0
        self.last_altitude = None
        self.last_depth_time = None

        # We want to fly to the hover height:
        self.map_height = 5.5

        # XY PID constants
        self.x_error = 0.0
        self.y_error = 0.0
        self.last_img_time = None
        self.last_x_error = 0.0
        self.last_y_error = 0.0
        self.x_error_dot = 0.0
        self.y_error_dot = 0.0
        self.map_angle = 0
        self.x_error_int = 0.0
        self.y_error_int = 0.0
        self.xy_int_limit = 4000.0
        self.xy_deadband = 8.0
        self.max_xy_force = 25.0
        self.k_xy_p = 0.3
        self.k_xy_d = 0.4
        self.k_xy_i = 0.002

        # Altitude PD constants (1 ,4):
        self.kz_p = 1.0
        self.kz_d = 4.0
        self.depth_grid_n = 3
        self.depth_margin = 0.30
        self.depth_percentile = 85.0
        self.max_downwards = 10.0
        self.max_upwards = 120

        self.k_roll = 1.0
        self.k_pitch = 1.0
        self.k_rollrate = 0.2
        self.k_pitchrate = 0.2
        self.k_yaw = -1.0
        self.k_yawrate = 0.25

        self.rate = rospy.Rate(50)

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        scale = 0.2
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = cv2.bitwise_not(cv2.bitwise_and(cv2.inRange(h, 0, 10), cv2.inRange(v, 170, 180)))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        M = cv2.moments(mask)
        if M["m00"] == 0:
            return

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        cx /= scale
        cy /= scale

        img_center_x = img.shape[1] / 2.0
        img_center_y = img.shape[0] / 2.0

        new_x_error = cx - img_center_x
        new_y_error = cy - img_center_y

        now = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()

        if self.last_img_time is None:
            self.x_error_dot = 0.0
            self.y_error_dot = 0.0
        else:
            dt = max(now - self.last_img_time, 1e-3)

            raw_x_dot = (new_x_error - self.last_x_error) / dt
            raw_y_dot = (new_y_error - self.last_y_error) / dt

            alpha_d = 0.25
            self.x_error_dot = alpha_d * raw_x_dot + (1.0 - alpha_d) * self.x_error_dot
            self.y_error_dot = alpha_d * raw_y_dot + (1.0 - alpha_d) * self.y_error_dot

        self.x_error = new_x_error
        self.y_error = new_y_error
        self.last_x_error = new_x_error
        self.last_y_error = new_y_error
        self.last_img_time = now
        self.map_angle = angle

        # vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # cv2.circle(vis, (int(cx * scale), int(cy * scale)), 3, (0, 0, 255), -1)
        # cv2.imshow("rect", vis)
        # cv2.waitKey(1)
            
    def imu_cb(self, msg):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        self.R = tft.quaternion_matrix(quat)[:3, :3].astype(np.float32)
        self.roll, self.pitch, self.yaw = tft.euler_from_quaternion(quat)
        self.map_angle = (np.pi /2) + self.yaw
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

        m = self.depth_margin
        n = self.depth_grid_n

        xs = np.linspace(int(m * w), int((1.0 - m) * w), n, dtype=np.int32)
        ys = np.linspace(int(m * h), int((1.0 - m) * h), n, dtype=np.int32)

        vals = []
        for y in ys:
            for x in xs:
                y0 = max(0, y - 1)
                y1 = min(h, y + 2)
                x0 = max(0, x - 1)
                x1 = min(w, x + 2)

                patch = depth[y0:y1, x0:x1].astype(np.float32)
                valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 20.0)]

                if valid.size > 0:
                    vals.append(float(np.mean(valid)))

        if not vals:
            return

        vals = np.asarray(vals, dtype=np.float32)
        z = float(np.percentile(vals, self.depth_percentile))

        now = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()

        if self.altitude is None:
            self.altitude = z
            self.last_altitude = z
            self.last_depth_time = now
            self.vertical_speed = 0.0
            return

        dt = max(now - self.last_depth_time, 1e-3)
        raw_vz = (z - self.last_altitude) / dt

        self.altitude = z
        self.vertical_speed = raw_vz

        self.last_altitude = z
        self.last_depth_time = now

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
                self.k_yaw * self.map_angle - self.k_yawrate * self.ang_vel[2]
            ], dtype=np.float32)

            torque_world = self.R.dot(torque_body)

            w.torque.x = float(torque_world[0])
            w.torque.y = float(torque_world[1])
            w.torque.z = float(torque_world[2])

            # deadband so we do not integrate tiny vision noise
            x_for_int = 0.0 if abs(self.x_error) < self.xy_deadband else self.x_error
            y_for_int = 0.0 if abs(self.y_error) < self.xy_deadband else self.y_error

            # leaky integrator: helps when wind state changes
            leak = 0.995
            self.x_error_int = leak * self.x_error_int + x_for_int * dt
            self.y_error_int = leak * self.y_error_int + y_for_int * dt

            self.x_error_int = float(np.clip(self.x_error_int, -self.xy_int_limit, self.xy_int_limit))
            self.y_error_int = float(np.clip(self.y_error_int, -self.xy_int_limit, self.xy_int_limit))
            
            # image positive x is body -y and image y is body +x?
            force_body = np.array([
                -self.k_xy_p * self.y_error
                - self.k_xy_d * self.y_error_dot
                - self.k_xy_i * self.y_error_int,

                -self.k_xy_p * self.x_error
                - self.k_xy_d * self.x_error_dot
                - self.k_xy_i * self.x_error_int,

                0.0
            ], dtype=np.float32)

            force_body[0] = np.clip(force_body[0], -self.max_xy_force, self.max_xy_force)
            force_body[1] = np.clip(force_body[1], -self.max_xy_force, self.max_xy_force)

            force_world = self.R.dot(force_body)

            w.force.x = float(force_world[0])
            w.force.y = float(force_world[1])

            thrust_z = self.mass * self.g

            if self.altitude is not None:
                z_target = self.map_height
                z_err = z_target - self.altitude

                z_correction = (self.kz_p * z_err - self.kz_d * self.vertical_speed)
                z_correction = float(np.clip(
                    z_correction,
                    -self.max_downwards,
                    self.max_upwards
                ))

                thrust_z += z_correction

            w.force.z = float(thrust_z)

            rospy.loginfo_throttle(
                0.2,
                f"alt={self.altitude} xerror={-self.y_error} y_error={self.x_error:.2f} angle={self.yaw:.3f}"
                f"F=({w.force.x:.1f}, {w.force.y:.1f}, {w.force.z:.1f})"
            )

            self.pub.publish(w)
            self.rate.sleep()


if __name__ == "__main__":
    DroneController().run()