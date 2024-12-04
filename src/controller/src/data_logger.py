import os
import rospy
import datetime
import numpy as np
from threading import Thread, Event
import rospkg
from PyQt5.QtCore import QObject, pyqtSlot
import cv2
import csv
from queue import Queue

class DataLogger(QObject):
    def __init__(self):
        super(DataLogger, self).__init__()
        self.logging = False
        self.save_queue = Queue()
        self.save_thread = None
        self.stop_event = Event()
        self.csv_file = None
        self.csv_writer = None

    def start_logging(self):
        if not self.logging:
            self.logging = True
            self.stop_event.clear()
            self.save_thread = Thread(target=self._save_data_loop, daemon=True)
            self.save_thread.start()
            rospy.loginfo("Data logging started.")

            # Initialize CSV file
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('controller')
            save_dir = os.path.join(package_path, 'logged_data')
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = os.path.join(save_dir, f"twist_data_{timestamp}.csv")

            try:
                self.csv_file = open(csv_filename, mode='w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                # Write CSV header
                self.csv_writer.writerow(['timestamp', 'image_filename', 'linear_x', 'angular_z'])
                rospy.loginfo(f"CSV file created at {csv_filename}")
            except Exception as e:
                rospy.logerr(f"Failed to create CSV file: {e}")
                self.logging = False

    def stop_logging(self):
        if self.logging:
            self.logging = False
            self.stop_event.set()
            self.save_thread.join()
            self.save_thread = None
            rospy.loginfo("Data logging stopped.")
            if self.csv_file:
                self.csv_file.close()
                rospy.loginfo("CSV file closed.")

    def _save_data_loop(self):
        while not self.stop_event.is_set():
            try:
                # Wait for data to save, with timeout to allow checking stop_event
                data = self.save_queue.get(timeout=0.1)
                img_array, linear_x, angular_z, timestamp = data

                # Define the directory to save the logged data
                rospack = rospkg.RosPack()
                package_path = rospack.get_path('controller')
                save_dir = os.path.join(package_path, 'logged_data', 'images')
                os.makedirs(save_dir, exist_ok=True)

                # Generate a timestamped filename
                img_filename = f"image_{timestamp}.png"
                img_path = os.path.join(save_dir, img_filename)

                # Convert the image array from RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Save the image as PNG
                cv2.imwrite(img_path, img_bgr)
                rospy.logdebug(f"Image saved to {img_path}")

                # Write the twist data to CSV
                if self.csv_writer:
                    self.csv_writer.writerow([timestamp, img_filename, linear_x, angular_z])
                    rospy.logdebug(f"Twist data logged for {img_filename}")

                self.save_queue.task_done()

            except Exception as e:
                # Handle empty queue or other exceptions
                continue

    @pyqtSlot(np.ndarray, float, float)
    def receive_data(self, img_array, linear_x, angular_z):
        if self.logging:
            if img_array is not None:
                # Generate a precise timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # Put the data into the save queue
                self.save_queue.put((img_array, linear_x, angular_z, timestamp))
                rospy.logdebug(f"Data queued for image_{timestamp}.png")
            else:
                rospy.logwarn("No image data available to log.")
