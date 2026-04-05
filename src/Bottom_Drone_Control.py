#!/usr/bin/env python3

import os
import cv2
import rospy
import weiweiOCR
import numpy as np
import tf.transformations as tft

from cv_bridge import CvBridge
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String

class DroneController:
    def __init__(self):
        rospy.init_node("bottom_drone_controller")

        self.bridge = CvBridge()

        self.TYPE_TO_LOCATION = {
            "SIZE": 1,
            "VICTIM": 2,
            "CRIME": 3,
            "TIME": 4,
            "PLACE": 5,
            "MOTIVE": 6,
            "WEAPON": 7,
            "BANDIT": 8,
        }
        self.TEAM_NAME = "Weidan"
        self.TEAM_PASS = "1234"

        self.save_dir = os.path.expanduser("~/Pictures/run_photos")
        os.makedirs(self.save_dir, exist_ok=True)

        self.no_clueboard_waypoints = {6, 8}  # sign_07 and sign_09
        self.saved_waypoints = set()
        self.timer_started = False
        self.results_submitted = False

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

        # XY PID
        self.x_error = 0.0
        self.y_error = 0.0
        self.last_img_time = None
        self.last_x_error = 0.0
        self.last_y_error = 0.0
        self.x_error_dot = 0.0
        self.y_error_dot = 0.0
        self.x_error_int = 0.0
        self.y_error_int = 0.0
        self.xy_int_limit = 4000.0
        self.xy_deadband = 0.5
        self.max_xy_force = 25.0
        self.k_xy_p = 0.5
        self.k_xy_d = 0.65
        self.k_xy_i = 0.0375

        # Z control
        self.kz_p = 6.0
        self.kz_d = 4.0
        self.depth_grid_n = 3
        self.depth_margin = 0.30
        self.depth_percentile = 50.0
        self.max_downwards = 20.0
        self.max_upwards = 20.0

        # Attitude / yaw control
        self.k_roll = 1.0
        self.k_pitch = 1.0
        self.k_rollrate = 0.2
        self.k_pitchrate = 0.2
        self.k_yaw = 1.0
        self.k_yawrate = 1.25

        # Camera geometry
        self.top_altitude = 11.9
        self.bottom_altitude = 1.0

        # Homographies
        self.H_img_to_field = None
        self.H_field_to_img = None

        # D2 position in image pixels
        self.curr_px = None
        self.curr_py = None

        # D2 position in global normalized field coords
        self.curr_u = None
        self.curr_v = None

        # Active target
        self.target_u = None
        self.target_v = None
        self.target_z = 0.30
        self.target_yaw = -np.pi / 2.0

        # Active target reprojected into current D1 image
        self.target_px = None
        self.target_py = None

        self.latest_view = None
        self.latest_sign_img = None
        self.latest_sign_stamp = rospy.Time(0)
        self.control_enabled = True

        def deg(x):
            return np.deg2rad(x)

        self.waypoints = [
            {"u": 0.006, "v": 0.952, "z": 0.30, "yaw": deg(-90),  "delay": 5.0},
            {"u": 0.029, "v": 0.322, "z": 0.30, "yaw": deg(-90),  "delay": 3.0},
            {"u": 0.080, "v": 0.192, "z": 0.30, "yaw": deg(-180), "delay": 3.0},
            {"u": 0.446, "v": 0.263, "z": 0.30, "yaw": deg(90),   "delay": 3.0},
            {"u": 0.454, "v": 0.937, "z": 0.30, "yaw": deg(-90),  "delay": 5.0},
            {"u": 0.724, "v": 0.781, "z": 0.30, "yaw": deg(-180), "delay": 5.0},
            {"u": 0.898, "v": 0.804, "z": 0.30, "yaw": deg(-90),  "delay": 2.0},
            {"u": 0.904, "v": 0.123, "z": 0.30, "yaw": deg(0),    "delay": 5.0},
            {"u": 0.832, "v": 0.184, "z": 1.8,  "yaw": deg(0),    "delay": 13.0},
            {"u": 0.648, "v": 0.228, "z": 0.30, "yaw": deg(0),    "delay": 5.0},
        ]

        self.goal_tol_global = 0.02
        self.goal_tol_z = 0.10
        self.goal_tol_yaw = np.deg2rad(10.0)

        self.wp_idx = 0
        self.state = "RUNNING"
        self.wait_end_time = None

        rospy.Subscriber("/D2/drone/camera1/image_raw", Image, self.sign_cam_cb, queue_size=1)

        cv2.namedWindow("view")

        self.pub = rospy.Publisher("/D2/force_cmd", Wrench, queue_size=10)
        self.score_tracker = rospy.Publisher("/score_tracker", String, queue_size=10)

        rospy.Subscriber("/D1/down_cam/image_raw", Image, self.image_cb, queue_size=1)
        rospy.Subscriber("/D2/imu", Imu, self.imu_cb, queue_size=1)
        rospy.Subscriber("/D2/down_cam/depth/image_raw", Image, self.depth_cb, queue_size=1)

        self.rate = rospy.Rate(50)
        self.set_target_waypoint(self.wp_idx)

    @staticmethod
    def wrap_pi(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def order_box(self, box):
        s = box.sum(axis=1)
        d = np.diff(box, axis=1).reshape(-1)
        return np.array([
            box[np.argmin(s)],
            box[np.argmin(d)],
            box[np.argmax(s)],
            box[np.argmax(d)],
        ], dtype=np.float32)

    def img_to_field(self, x, y):
        if self.H_img_to_field is None:
            return None
        pt = np.array([[[x, y]]], dtype=np.float32)
        uv = cv2.perspectiveTransform(pt, self.H_img_to_field)[0, 0]
        scale = self.top_altitude / max(self.top_altitude - self.bottom_altitude, 1e-3)
        u = 0.5 + (uv[0] - 0.5) * scale
        v = 0.5 + (uv[1] - 0.5) * scale
        return float(u), float(v)

    def field_to_img(self, u, v):
        if self.H_field_to_img is None:
            return None
        scale = self.top_altitude / max(self.top_altitude - self.bottom_altitude, 1e-3)
        uu = 0.5 + (u - 0.5) / scale
        vv = 0.5 + (v - 0.5) / scale
        pt = np.array([[[uu, vv]]], dtype=np.float32)
        xy = cv2.perspectiveTransform(pt, self.H_field_to_img)[0, 0]
        return float(xy[0]), float(xy[1])

    def set_target(self, u, v, z, yaw):
        self.target_u = float(u)
        self.target_v = float(v)
        self.target_z = float(z)
        self.target_yaw = float(self.wrap_pi(yaw))

    def set_target_waypoint(self, idx):
        wp = self.waypoints[idx]
        self.set_target(wp["u"], wp["v"], wp["z"], wp["yaw"])
        self.reset_xy_pid()
        print(
            "target -> waypoint {} = ({:.4f}, {:.4f}), z = {:.2f}, yaw = {:.1f} deg".format(
                idx + 1, self.target_u, self.target_v, self.target_z, np.rad2deg(self.target_yaw)
            )
        )

    def reset_xy_pid(self):
        self.x_error = 0.0
        self.y_error = 0.0
        self.last_x_error = 0.0
        self.last_y_error = 0.0
        self.x_error_dot = 0.0
        self.y_error_dot = 0.0
        self.x_error_int = 0.0
        self.y_error_int = 0.0
        self.last_img_time = None

    def publish_score(self, location, clue="NA", pause=0.75):
        clue = str(clue).strip().replace(",", "")
        msg = f"{self.TEAM_NAME},{self.TEAM_PASS},{int(location)},{clue}"
        rospy.loginfo("score_tracker <- %s", msg)
        self.score_tracker.publish(msg)
        rospy.sleep(pause)

    def start_timer(self):
        if self.timer_started:
            return
        self.publish_score(0, "NA", pause=1.0)
        self.timer_started = True

    def stop_timer(self):
        self.publish_score(-1, "NA", pause=1.0)
        
    def normalize_type(self, text):
        return str(text).strip().upper().replace(",", "")

    def normalize_clue(self, text):
        return str(text).strip().upper().replace(",", "")

    def submit_points(self):
        if self.results_submitted:
            return

        for idx in sorted(self.saved_waypoints):
            if idx in self.no_clueboard_waypoints:
                continue

            filename = os.path.join(self.save_dir, f"sign_{idx + 1:02d}.png")
            img = cv2.imread(filename)
            if img is None:
                rospy.logwarn("Could not read %s", filename)
                continue

            try:
                result = weiweiOCR.read_sign(img)
            except Exception as e:
                rospy.logwarn("OCR failed on %s: %s", filename, str(e))
                continue

            if result is None or len(result) != 2:
                rospy.logwarn("OCR returned invalid result for %s: %s", filename, str(result))
                continue

            sign_type, clue = result
            sign_type = self.normalize_type(sign_type)
            clue = self.normalize_clue(clue)

            if sign_type not in self.TYPE_TO_LOCATION:
                rospy.logwarn("Unknown sign type '%s' from %s", sign_type, filename)
                continue

            self.publish_score(self.TYPE_TO_LOCATION[sign_type], clue)

        self.stop_timer()
        self.results_submitted = True

    def advance_waypoint_or_finish(self):
        self.wp_idx += 1
        if self.wp_idx >= len(self.waypoints):
            self.wp_idx = len(self.waypoints) - 1
            self.state = "DONE"
            self.submit_points()
            print("mission complete")
        else:
            self.set_target_waypoint(self.wp_idx)
            self.state = "RUNNING"

    def imu_cb(self, msg):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        self.R = tft.quaternion_matrix(quat)[:3, :3].astype(np.float32)
        self.roll, self.pitch, self.yaw = tft.euler_from_quaternion(quat)
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

        z = float(np.percentile(np.asarray(vals, dtype=np.float32), self.depth_percentile))
        now = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()

        if self.altitude is None:
            self.altitude = z
            self.last_altitude = z
            self.last_depth_time = now
            self.vertical_speed = 0.0
            return

        dt = max(now - self.last_depth_time, 1e-3)
        self.altitude = z
        self.vertical_speed = (z - self.last_altitude) / dt
        self.last_altitude = z
        self.last_depth_time = now

    def update_mission_state(self):
        if self.state == "DONE":
            return

        if self.curr_u is None or self.curr_v is None or self.target_u is None:
            return

        now = rospy.Time.now()

        if self.state == "WAITING":
            if now >= self.wait_end_time:
                self.save_sign_image(self.wp_idx)
                self.advance_waypoint_or_finish()
            return

        xy_err = np.hypot(self.target_u - self.curr_u, self.target_v - self.curr_v)
        z_err = 0.0 if self.altitude is None else abs(self.target_z - self.altitude)
        yaw_err = abs(self.wrap_pi(self.target_yaw - self.yaw))

        xy_ok = xy_err < self.goal_tol_global
        z_ok = (self.altitude is None) or (z_err < self.goal_tol_z)
        yaw_ok = yaw_err < self.goal_tol_yaw

        if xy_ok and z_ok and yaw_ok:
            delay = float(self.waypoints[self.wp_idx].get("delay", 0.0))
            if delay > 0.0:
                self.state = "WAITING"
                self.wait_end_time = now + rospy.Duration(delay)
                print("waypoint {} reached, waiting {:.2f}s".format(self.wp_idx + 1, delay))
            else:
                self.advance_waypoint_or_finish()

    def sign_cam_cb(self, msg):
        try:
            self.latest_sign_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_sign_stamp = msg.header.stamp
        except Exception as e:
            rospy.logwarn_throttle(1.0, "sign_cam_cb failed: %s", str(e))

    def save_sign_image(self, idx):
        if idx in self.saved_waypoints:
            return

        if self.latest_sign_img is None:
            rospy.logwarn("No sign image available to save for waypoint %d", idx + 1)
            return

        if self.latest_sign_stamp != rospy.Time(0):
            age = (rospy.Time.now() - self.latest_sign_stamp).to_sec()
            if age > 1.0:
                rospy.logwarn("Saving stale sign image for waypoint %d (age %.2fs)", idx + 1, age)

        filename = os.path.join(self.save_dir, f"sign_{idx + 1:02d}.png")
        ok = cv2.imwrite(filename, self.latest_sign_img.copy())

        if ok:
            self.saved_waypoints.add(idx)
            rospy.loginfo("saved %s", filename)
        else:
            rospy.logwarn("failed to save %s", filename)

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        field_mask = cv2.bitwise_not(
            cv2.bitwise_and(cv2.inRange(h, 0, 1), cv2.inRange(v, 175, 180))
        )

        contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = img.copy()

        if not contours:
            self.latest_view = vis
            return

        field = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(field)
        box = self.order_box(cv2.boxPoints(rect).astype(np.float32))

        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self.H_img_to_field = cv2.getPerspectiveTransform(box, dst)
        self.H_field_to_img = cv2.getPerspectiveTransform(dst, box)

        purple_mask = cv2.bitwise_and(cv2.inRange(h, 130, 145), cv2.inRange(s, 240, 255))
        purple_mask = cv2.bitwise_and(purple_mask, field_mask)

        contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.curr_px = self.curr_py = self.curr_u = self.curr_v = None

        if contours:
            purple = max(contours, key=cv2.contourArea)
            M = cv2.moments(purple)
            if M["m00"] != 0:
                self.curr_px = M["m10"] / M["m00"]
                self.curr_py = M["m01"] / M["m00"]
                out = self.img_to_field(self.curr_px, self.curr_py)
                if out is not None:
                    self.curr_u, self.curr_v = out

        self.target_px = self.target_py = None
        if self.target_u is not None and self.target_v is not None:
            out = self.field_to_img(self.target_u, self.target_v)
            if out is not None:
                self.target_px, self.target_py = out

        if self.curr_px is not None and self.target_px is not None:
            self.x_error = self.target_px - self.curr_px
            self.y_error = self.target_py - self.curr_py

            now = rospy.Time.now()
            if self.last_img_time is not None:
                dt = max((now - self.last_img_time).to_sec(), 1e-3)
                self.x_error_dot = (self.x_error - self.last_x_error) / dt
                self.y_error_dot = (self.y_error - self.last_y_error) / dt

            self.last_img_time = now
            self.last_x_error = self.x_error
            self.last_y_error = self.y_error
        else:
            self.x_error_dot = 0.0
            self.y_error_dot = 0.0

        cv2.polylines(vis, [box.astype(np.int32)], True, (0, 255, 0), 2)

        if self.curr_px is not None:
            cv2.circle(vis, (int(self.curr_px), int(self.curr_py)), 6, (0, 0, 255), -1)

        if self.target_px is not None:
            cv2.circle(vis, (int(self.target_px), int(self.target_py)), 7, (0, 255, 255), -1)

        for i, wp in enumerate(self.waypoints):
            out = self.field_to_img(wp["u"], wp["v"])
            if out is None:
                continue

            px, py = out
            color = (0, 255, 255) if i == self.wp_idx else (255, 255, 0)
            radius = 7 if i == self.wp_idx else 4

            cv2.circle(vis, (int(px), int(py)), radius, color, -1)
            cv2.putText(
                vis, str(i + 1), (int(px) + 6, int(py) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        cv2.putText(
            vis, f"MISSION: {self.state}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        cv2.putText(
            vis, f"WAYPOINT: {self.wp_idx + 1}/{len(self.waypoints)}",
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        target_text = "TARGET: (--, --)"
        if self.target_u is not None and self.target_v is not None:
            target_text = f"TARGET: ({self.target_u:.3f}, {self.target_v:.3f})"

        cv2.putText(
            vis, target_text,
            (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.putText(
            vis,
            "z: {:.2f} -> {:.2f}".format(
                self.altitude if self.altitude is not None else -1.0,
                self.target_z
            ),
            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.putText(
            vis,
            "yaw: {:.1f} -> {:.1f} deg".format(
                np.rad2deg(self.yaw),
                np.rad2deg(self.target_yaw)
            ),
            (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        if self.state == "WAITING" and self.wait_end_time is not None:
            remaining = max(0.0, (self.wait_end_time - rospy.Time.now()).to_sec())
            cv2.putText(
                vis, f"wait remaining: {remaining:.2f}s",
                (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        self.latest_view = vis

    def run(self):
        last_time = rospy.Time.now()
        rospy.sleep(1.0)

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = max((now - last_time).to_sec(), 1e-3)
            last_time = now

            self.update_mission_state()

            if self.latest_view is not None:
                cv2.imshow("view", self.latest_view)
                cv2.waitKey(1)

            w = Wrench()

            yaw_err = self.wrap_pi(self.target_yaw - self.yaw)
            torque_body = np.array([
                -self.k_roll * self.roll - self.k_rollrate * self.ang_vel[0],
                -self.k_pitch * self.pitch - self.k_pitchrate * self.ang_vel[1],
                self.k_yaw * yaw_err - self.k_yawrate * self.ang_vel[2]
            ], dtype=np.float32)
            torque_world = self.R.dot(torque_body)

            w.torque.x = float(torque_world[0])
            w.torque.y = float(torque_world[1])
            w.torque.z = float(torque_world[2])

            if self.control_enabled and self.curr_px is not None and self.target_px is not None:
                x_for_int = 0.0 if abs(self.x_error) < self.xy_deadband else self.x_error
                y_for_int = 0.0 if abs(self.y_error) < self.xy_deadband else self.y_error

                leak = 0.995
                self.x_error_int = leak * self.x_error_int + x_for_int * dt
                self.y_error_int = leak * self.y_error_int + y_for_int * dt

                self.x_error_int = float(np.clip(self.x_error_int, -self.xy_int_limit, self.xy_int_limit))
                self.y_error_int = float(np.clip(self.y_error_int, -self.xy_int_limit, self.xy_int_limit))

                ux = (
                    self.k_xy_p * self.x_error +
                    self.k_xy_d * self.x_error_dot +
                    self.k_xy_i * self.x_error_int
                )
                uy = (
                    self.k_xy_p * self.y_error +
                    self.k_xy_d * self.y_error_dot +
                    self.k_xy_i * self.y_error_int
                )

                w.force.x = float(np.clip(-ux, -self.max_xy_force, self.max_xy_force))
                w.force.y = float(np.clip(+uy, -self.max_xy_force, self.max_xy_force))
            else:
                w.force.x = 0.0
                w.force.y = 0.0

            thrust_z = self.mass * self.g
            if self.altitude is not None:
                z_err = self.target_z - self.altitude
                z_correction = self.kz_p * z_err - self.kz_d * self.vertical_speed
                thrust_z += float(np.clip(z_correction, -self.max_downwards, self.max_upwards))

            w.force.z = float(thrust_z)
            self.pub.publish(w)
            self.rate.sleep()


if __name__ == "__main__":
    DroneController().run()