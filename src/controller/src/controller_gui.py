#!/usr/bin/env python3

import sys
import os
import rospkg
import rospy
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ControllerGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(ControllerGUI, self).__init__()

        # Initialize ROS node
        rospy.init_node('controller_gui_node', anonymous=True)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Get the path to the 'controller' package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('controller')

        # Construct the full path to 'developer_tools.ui'
        ui_file = os.path.join(package_path, 'developer_tools.ui')  # Adjust if it's in a subdirectory

        # Load the UI file
        if not os.path.exists(ui_file):
            rospy.logerr(f"UI file not found: {ui_file}")
            sys.exit(1)
        else:
            uic.loadUi(ui_file, self)

        # Make movement buttons checkable
        self.move_forward.setCheckable(True)
        self.move_backward.setCheckable(True)
        self.move_left.setCheckable(True)
        self.move_right.setCheckable(True)

        # Set 'Raw' as the default option in mainCombo
        index = self.mainCombo.findText("Raw")
        if index != -1:
            self.mainCombo.setCurrentIndex(index)

        # Set up publishers
        self.pub_cmd_vel = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)

        # Connect UI elements to functions
        self.move_forward.clicked.connect(self.toggle_move_forward)
        self.move_backward.clicked.connect(self.toggle_move_backward)
        self.move_left.clicked.connect(self.toggle_move_left)
        self.move_right.clicked.connect(self.toggle_move_right)
        self.auto_drive_toggle.clicked.connect(self.auto_drive_toggle_function)
        self.saveImage.clicked.connect(self.save_image_function)

        # Initialize set to keep track of pressed keys
        self.pressed_keys = set()

        # Movement flags controlled by buttons
        self.button_move_forward = False
        self.button_move_backward = False
        self.button_move_left = False
        self.button_move_right = False

        # Start a timer to call publish_movement at regular intervals
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.publish_movement)
        self.timer.start(100)  # Every 100 ms (10 Hz)

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback)

        # Ensure the window can accept focus and receive key events
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    # Toggle functions for GUI buttons
    def toggle_move_forward(self):
        self.button_move_forward = self.move_forward.isChecked()
        if self.button_move_forward:
            self.move_forward.setStyleSheet("background-color: green")
        else:
            self.move_forward.setStyleSheet("")
        # rospy.loginfo(f"Move Forward: {'On' if self.button_move_forward else 'Off'}")

    def toggle_move_backward(self):
        self.button_move_backward = self.move_backward.isChecked()
        if self.button_move_backward:
            self.move_backward.setStyleSheet("background-color: green")
        else:
            self.move_backward.setStyleSheet("")
        # rospy.loginfo(f"Move Backward: {'On' if self.button_move_backward else 'Off'}")

    def toggle_move_left(self):
        self.button_move_left = self.move_left.isChecked()
        if self.button_move_left:
            self.move_left.setStyleSheet("background-color: green")
        else:
            self.move_left.setStyleSheet("")
        # rospy.loginfo(f"Move Left: {'On' if self.button_move_left else 'Off'}")

    def toggle_move_right(self):
        self.button_move_right = self.move_right.isChecked()
        if self.button_move_right:
            self.move_right.setStyleSheet("background-color: green")
        else:
            self.move_right.setStyleSheet("")
        # rospy.loginfo(f"Move Right: {'On' if self.button_move_right else 'Off'}")

    # Keyboard event handlers
    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            if key in [QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D]:
                self.pressed_keys.add(key)
                # rospy.logdebug(f"Key Pressed: {QtCore.Qt.keyToString(key)}")

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            if key in [QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D]:
                self.pressed_keys.discard(key)
                # rospy.logdebug(f"Key Released: {QtCore.Qt.keyToString(key)}")

    # Function to publish movement commands
    def publish_movement(self):
        twist = Twist()

        # Keyboard-controlled movement
        if QtCore.Qt.Key_W in self.pressed_keys:
            twist.linear.x += 5.0  # Move forward
        if QtCore.Qt.Key_S in self.pressed_keys:
            twist.linear.x -= 5.0  # Move backward
        if QtCore.Qt.Key_A in self.pressed_keys:
            twist.angular.z += 3.0  # Turn left
        if QtCore.Qt.Key_D in self.pressed_keys:
            twist.angular.z -= 3.0  # Turn right

        # Button-controlled movement
        if self.button_move_forward:
            twist.linear.x += 1.0
        if self.button_move_backward:
            twist.linear.x -= 1.0
        if self.button_move_left:
            twist.angular.z += 1.0
        if self.button_move_right:
            twist.angular.z -= 1.0

        # Publish the twist message
        self.pub_cmd_vel.publish(twist)
        # rospy.logdebug(f"Published Twist: linear.x={twist.linear.x}, angular.z={twist.angular.z}")

    def auto_drive_toggle_function(self):
        # Implement the logic to toggle auto-drive mode
        pass

    def save_image_function(self):
        # Implement the logic to save images from the robot's camera
        pass

    # Image callback
    def image_callback(self, msg):
        if self.mainCombo.currentText() != "Raw":
            return
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert the image to RGB format
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Get image dimensions
            height, width, channel = cv_image_rgb.shape
            bytes_per_line = 3 * width

            # Convert to QImage for mainfeed
            qt_image = QtGui.QImage(cv_image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

            # Scale the image to fit the QLabel while maintaining aspect ratio
            scaled_image = qt_image.scaled(self.mainfeed.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

            # Set the pixmap of the QLabel
            self.mainfeed.setPixmap(QtGui.QPixmap.fromImage(scaled_image))

            # ---- Billboard Processing ----

            # Create a mask for pixels with:
            # Blue >= 99, Green <= 100, Red <= 100
            blue_channel = cv_image_rgb[:, :, 2]
            green_channel = cv_image_rgb[:, :, 1]
            red_channel = cv_image_rgb[:, :, 0]

            mask = (blue_channel >= 99) & (green_channel <= 100) & (red_channel <= 100)

            # Create a binary (black and white) image based on the mask
            binary_image = np.where(mask, 255, 0).astype(np.uint8)

            # Convert the binary image to RGB format for display
            binary_image_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

            # Convert to QImage for billboard
            qt_billboard_image = QtGui.QImage(binary_image_rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888)

            # Scale the image to fit the billboard QLabel while maintaining aspect ratio
            scaled_billboard_image = qt_billboard_image.scaled(self.billboard.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

            # Set the pixmap of the billboard QLabel
            self.billboard.setPixmap(QtGui.QPixmap.fromImage(scaled_billboard_image))

            # ---- Billboard Indicator ----

            # Check if any pixels meet the condition
            if np.any(mask):
                # Set the billboard indicator to green
                self.label_billboard_indicator.setStyleSheet("""
                    QLabel {
                        background-color: green;
                        border-radius: 10px;
                    }
                """)
            else:
                # Set the billboard indicator to red
                self.label_billboard_indicator.setStyleSheet("""
                    QLabel {
                        background-color: red;
                        border-radius: 10px;
                    }
                """)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")


if __name__ == '__main__':
    # rospy.init_node('controller_gui_node', anonymous=True)  # Already initialized in __init__
    app = QtWidgets.QApplication(sys.argv)
    window = ControllerGUI()
    window.show()
    try:
        sys.exit(app.exec_())
    except rospy.ROSInterruptException:
        pass
