# bismillah_sequence.py

#!/usr/bin/env python3

import rospy
from PyQt5.QtCore import QObject, pyqtSignal
import time

class BismillahSequence(QObject):
    # Define signals to communicate with the main GUI thread
    set_linear = pyqtSignal(float)       # Emitted to set linear speed
    set_angular = pyqtSignal(float)      # Emitted to set angular speed
    stop_motion_signal = pyqtSignal()    # Emitted to stop motion
    request_predict = pyqtSignal()       # Emitted to request prediction
    teleport_signal = pyqtSignal(dict)    # Emitted to request teleportation

    def __init__(self, teleport_handler):
        super(BismillahSequence, self).__init__()
        self.teleport_handler = teleport_handler
        self._is_running = False

    def run_sequence(self):
        if self._is_running:
            rospy.logwarn("Bismillah sequence is already running.")
            return
        self._is_running = True
        self._sequence()

    def _sequence(self):
        try:
            # Step 1: Move forward
            self.set_linear.emit(2.7)
            self.set_angular.emit(0.0)
            rospy.loginfo("Step 1: Move forward.")
            time.sleep(0.5)
            self.stop_motion_signal.emit()

            
            # Step 2: Turn left
            self.set_linear.emit(0.0)
            self.set_angular.emit(2.0)
            rospy.loginfo("Step 2: Turn left.")
            time.sleep(1.0)
            self.stop_motion_signal.emit()

            # Step 3: Call predict function
            self.request_predict.emit()
            rospy.loginfo("Step 3: Predict.")
            time.sleep(2)
            
            # Step 4: Turn right
            self.set_linear.emit(0.0)
            self.set_angular.emit(-2.1)
            rospy.loginfo("Step 4: Turn right.")
            time.sleep(0.92)
            self.stop_motion_signal.emit()

            # Step 5: Move forward
            self.set_linear.emit(3.0)
            self.set_angular.emit(0.0)
            rospy.loginfo("Step 5: Move forward.")
            time.sleep(2.6)
            self.stop_motion_signal.emit()

            # Step 6: Nudge right
            self.set_linear.emit(0.0)
            self.set_angular.emit(-2.1)
            rospy.loginfo("Step 4: Turn right.")
            time.sleep(0.6)
            self.stop_motion_signal.emit()

            # Step 7: Call predict function
            self.request_predict.emit()
            rospy.loginfo("Step 7: Predict.")

            # Wait 2 seconds
            time.sleep(2)

            # Step 8: Teleport to first specified position
            teleport_pose2 = {
                'position': {'x': 0.5527487156990568, 'y': -0.12600472106246535, 'z': 0.04000056025622947},
                'orientation': {'x': 2.448884472570069e-07, 'y': 1.278603394462025e-07, 'z': -0.7071178631556455, 'w': -0.7070956990437133}
            }
            self.teleport_signal.emit(teleport_pose2)
            rospy.loginfo("Step 8: Teleport to position 1.")
            time.sleep(2)

            # Step 9: Move backward
            self.set_linear.emit(-3.0)
            self.set_angular.emit(0.0)
            rospy.loginfo("Step 9: Move backward.")
            time.sleep(0.7)
            self.stop_motion_signal.emit()

            # Step 10: Turn right
            self.set_linear.emit(0.0)
            self.set_angular.emit(-2.0)
            rospy.loginfo("Step 10: Turn right.")
            time.sleep(0.7)
            self.stop_motion_signal.emit()

            # Step 11: Call predict function
            self.request_predict.emit()
            rospy.loginfo("Step 11: Predict.")

            # Wait 2 seconds
            time.sleep(2)

            # Step 12: Teleport to second specified position
            teleport_pose1 = {
                'position': {'x': -4.008629252948557, 'y': 0.4038460916553116, 'z': 0.04000060474749287},
                'orientation': {'x': 7.154773655035404e-07, 'y': -1.501085552816514e-06, 'z': -0.4271649858420221, 'w': -0.9041736972882035}
            }
            self.teleport_signal.emit(teleport_pose1)
            rospy.loginfo("Step 12: Teleport to position 2.")
            time.sleep(2)

            # Step 13: Move forward
            self.set_linear.emit(3.0)
            self.set_angular.emit(0.0)
            rospy.loginfo("Step 13: Move forward.")
            time.sleep(1.3)
            self.stop_motion_signal.emit()

            # Step 14: Turn left
            self.set_linear.emit(0.0)
            self.set_angular.emit(2.0)
            rospy.loginfo("Step 14: Turn left.")
            time.sleep(2.3)
            self.stop_motion_signal.emit()

            # Step 15: Call predict function
            self.request_predict.emit()
            rospy.loginfo("Step 15: Predict.")

            # Wait 2 seconds
            time.sleep(5)

            # Step 16: Teleport to final specified position
            teleport_pose3 = {
                'position': {'x': -4.063470518828452, 'y': -2.2564473800170513, 'z': 0.04000018393550679},
                'orientation': {'x': -2.000162652701596e-07, 'y': -5.830896379528323e-07, 'z': -0.3377707635826839, 'w': -0.9412284054725457}
            }
            self.teleport_signal.emit(teleport_pose3)
            rospy.loginfo("Step 16: Teleport to final position.")

            # Step 17: Call predict function
            self.request_predict.emit()
            rospy.loginfo("Step 17: Predict.")

            rospy.loginfo("Bismillah sequence completed.")
        finally:
            self._is_running = False
