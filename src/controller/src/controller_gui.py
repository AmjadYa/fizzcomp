#!/usr/bin/env python3

import sys
import os
import rospkg
import rospy
from PyQt5 import QtWidgets, uic, QtCore
from geometry_msgs.msg import Twist

class ControllerGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(ControllerGUI, self).__init__()

        # Initialize ROS node
        rospy.init_node('controller_gui_node', anonymous=True)

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

        # Ensure the window can accept focus and receive key events
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    # Toggle functions for GUI buttons
    def toggle_move_forward(self):
        self.button_move_forward = self.move_forward.isChecked()
        if self.button_move_forward:
            self.move_forward.setStyleSheet("background-color: green")
        else:
            self.move_forward.setStyleSheet("")
        rospy.loginfo(f"Move Forward: {'On' if self.button_move_forward else 'Off'}")

    def toggle_move_backward(self):
        self.button_move_backward = self.move_backward.isChecked()
        if self.button_move_backward:
            self.move_backward.setStyleSheet("background-color: green")
        else:
            self.move_backward.setStyleSheet("")
        rospy.loginfo(f"Move Backward: {'On' if self.button_move_backward else 'Off'}")

    def toggle_move_left(self):
        self.button_move_left = self.move_left.isChecked()
        if self.button_move_left:
            self.move_left.setStyleSheet("background-color: green")
        else:
            self.move_left.setStyleSheet("")
        rospy.loginfo(f"Move Left: {'On' if self.button_move_left else 'Off'}")

    def toggle_move_right(self):
        self.button_move_right = self.move_right.isChecked()
        if self.button_move_right:
            self.move_right.setStyleSheet("background-color: green")
        else:
            self.move_right.setStyleSheet("")
        rospy.loginfo(f"Move Right: {'On' if self.button_move_right else 'Off'}")

    # Keyboard event handlers
    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            if key in [QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D]:
                self.pressed_keys.add(key)
                rospy.logdebug(f"Key Pressed: {QtCore.Qt.keyToString(key)}")

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            if key in [QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D]:
                self.pressed_keys.discard(key)
                rospy.logdebug(f"Key Released: {QtCore.Qt.keyToString(key)}")

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
        rospy.logdebug(f"Published Twist: linear.x={twist.linear.x}, angular.z={twist.angular.z}")

    def auto_drive_toggle_function(self):
        # Implement the logic to toggle auto-drive mode
        pass

    def save_image_function(self):
        # Implement the logic to save images from the robot's camera
        pass

if __name__ == '__main__':
    rospy.init_node('controller_gui_node', anonymous=True)  # Initialize ROS node here to avoid issues with QApplication
    app = QtWidgets.QApplication(sys.argv)
    window = ControllerGUI()
    window.show()
    sys.exit(app.exec_())
