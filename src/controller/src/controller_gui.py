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

        # Set up publishers
        self.pub_cmd_vel = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)

        # Connect UI elements to functions
        self.move_forward.clicked.connect(self.move_forward_function)
        self.move_backward.clicked.connect(self.move_backward_function)
        self.move_left.clicked.connect(self.move_left_function)
        self.move_right.clicked.connect(self.move_right_function)
        self.auto_drive_toggle.clicked.connect(self.auto_drive_toggle_function)
        self.saveImage.clicked.connect(self.save_image_function)

        # Initialize set to keep track of pressed keys
        self.pressed_keys = set()

        # Start a timer to call publish_movement at regular intervals
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.publish_movement)
        self.timer.start(100)  # Every 100 ms (10 Hz)

        # Ensure the window can accept focus and receive key events
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            self.pressed_keys.add(key)
            # Optionally, you can call publish_movement() here if you want immediate response
            # self.publish_movement()

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            self.pressed_keys.discard(key)
            # Optionally, stop movement immediately upon key release
            # self.publish_movement()

    def publish_movement(self):
        twist = Twist()

        # Check which keys are pressed and set the twist accordingly
        if QtCore.Qt.Key_W in self.pressed_keys:
            twist.linear.x += 1.0  # Adjust speed as needed
        if QtCore.Qt.Key_S in self.pressed_keys:
            twist.linear.x -= 1.0
        if QtCore.Qt.Key_A in self.pressed_keys:
            twist.angular.z += 1.0
        if QtCore.Qt.Key_D in self.pressed_keys:
            twist.angular.z -= 1.0

        # Publish the twist message
        self.pub_cmd_vel.publish(twist)

    def move_forward_function(self):
        twist = Twist()
        twist.linear.x = 1.0  # Adjust the speed as needed
        self.pub_cmd_vel.publish(twist)

    def move_backward_function(self):
        twist = Twist()
        twist.linear.x = -1.0  # Adjust the speed as needed
        self.pub_cmd_vel.publish(twist)

    def move_left_function(self):
        twist = Twist()
        twist.angular.z = 1.0  # Adjust the angular speed as needed
        self.pub_cmd_vel.publish(twist)

    def move_right_function(self):
        twist = Twist()
        twist.angular.z = -1.0  # Adjust the angular speed as needed
        self.pub_cmd_vel.publish(twist)

    def auto_drive_toggle_function(self):
        # Implement the logic to toggle auto-drive mode
        pass

    def save_image_function(self):
        # Implement the logic to save images from the robot's camera
        pass

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ControllerGUI()
    window.show()
    sys.exit(app.exec_())
