import os
import rospy
import datetime
import numpy as np
from threading import Thread, Event
import rospkg
from PyQt5.QtCore import QObject, pyqtSlot

class DataLogger(QObject):
    def __init__(self):
        super(DataLogger, self).__init__()
        self.data_log = []
        self.logging = False
        self.thread = None
        self.stop_event = Event()

    def start_logging(self):
        if not self.logging:
            self.logging = True
            self.stop_event.clear()
            self.thread = Thread(target=self._log_data_loop)
            self.thread.start()
            rospy.loginfo("Data logging started.")

    def stop_logging(self):
        if self.logging:
            self.logging = False
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            rospy.loginfo("Data logging stopped.")
            self.save_logged_data()

    def _log_data_loop(self):
        rate = rospy.Rate(10)  # 10 Hz (every 100 ms)
        while not self.stop_event.is_set():
            rate.sleep()  # Wait for next interval
            # Logging is handled via slots; nothing needed here

    @pyqtSlot(np.ndarray, float, float)
    def receive_data(self, img_array, linear_x, angular_z):
        if self.logging:
            if img_array is not None:
                self.data_log.append((img_array, linear_x, angular_z))
                rospy.logdebug(f"Logged image and twist: linear.x={linear_x}, angular.z={angular_z}")
            else:
                rospy.logwarn("No image data available to log.")

    def save_logged_data(self):
        try:
            # Define the directory to save the logged data
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('controller')
            save_dir = os.path.join(package_path, 'logged_data')
            os.makedirs(save_dir, exist_ok=True)

            # Generate a timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(save_dir, f"data_log_{timestamp}.npz")

            # Convert the log to structured arrays
            images = [entry[0] for entry in self.data_log]
            linear_twists = [entry[1] for entry in self.data_log]
            angular_twists = [entry[2] for entry in self.data_log]

            # Save as a compressed NumPy file
            np.savez_compressed(file_path, images=images, linear_twists=linear_twists, angular_twists=angular_twists)
            rospy.loginfo(f"Logged data saved to {file_path}")
        except Exception as e:
            rospy.logerr(f"Error saving logged data: {e}")
