#!/usr/bin/env python3
import subprocess
import signal
import rospy
import os

class CompetitionStart:
    def __init__(self):
        rospy.init_node("competition_start")

        self.startup_delay = rospy.get_param("~startup_delay", 25.0)


        self.top_cmd = ["rosrun", "my_controller", "Top_Drone_Control.py"]
        self.bottom_cmd = ["rosrun", "my_controller", "Bottom_Drone_Control.py"]

        scripts_dir = os.path.expanduser(
            "~/ros_ws/src/2025_competition/enph353/enph353_utils/scripts"
        )

        rospy.loginfo("starting score tracker")
        self.score_tracker_proc = subprocess.Popen(
            ["./score_tracker.py"],
            cwd=scripts_dir
        )

        rospy.loginfo("starting top drone controller")
        self.top_proc = subprocess.Popen(self.top_cmd)

        rospy.loginfo("waiting %.1f s before starting bottom drone", self.startup_delay)
        rospy.sleep(self.startup_delay)

        rospy.loginfo("starting bottom drone controller")
        self.bottom_proc = subprocess.Popen(self.bottom_cmd)

        rospy.on_shutdown(self.shutdown)
        rospy.spin()

    def shutdown(self):
        for proc in [
            getattr(self, "bottom_proc", None),
            getattr(self, "top_proc", None),
            getattr(self, "score_tracker_proc", None),
        ]:
            if proc is None:
                continue
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

if __name__ == "__main__":
    CompetitionStart()