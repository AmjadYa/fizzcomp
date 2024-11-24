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
import datetime

from PyQt5.QtCore import pyqtSignal

class ControllerGUI(QtWidgets.QMainWindow):
    # Define a signal that carries the processed image and billCombo selection
    image_update_signal = pyqtSignal(np.ndarray, str)

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

        # Set 'Raw' as the default option in billCombo
        bill_index = self.billCombo.findText("Raw")
        if bill_index != -1:
            self.billCombo.setCurrentIndex(bill_index)

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

        # Connect the image update signal to the update_billboard slot
        self.image_update_signal.connect(self.update_billboard)

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

    def save_image_function(self):
        """
        Saves the current image displayed on the billboard QLabel to a file.
        """
        try:
            # Retrieve the current pixmap from the billboard
            pixmap = self.billboard.pixmap()
            if pixmap:
                # Get the path to the 'controller' package
                rospack = rospkg.RosPack()
                package_path = rospack.get_path('controller')

                # Define the directory to save images
                save_dir = os.path.join(package_path, 'saved_images')
                os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

                # Generate a timestamped filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"billboard_{timestamp}.png"
                file_path = os.path.join(save_dir, filename)

                # Save the pixmap to the file
                if not pixmap.save(file_path):
                    rospy.logerr(f"Failed to save image to {file_path}")
                else:
                    rospy.loginfo(f"Image saved to {file_path}")
            else:
                rospy.logwarn("No image to save on the billboard.")
        except Exception as e:
            rospy.logerr(f"Error saving image: {e}")

    def auto_drive_toggle_function(self):
        # Implement 
        pass

    # Helper function to outline the largest contour on a binary image
    def outline_largest_contour(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None  # No contours found

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the largest contour on the binary image
        outlined_image = binary_image.copy()
        cv2.drawContours(outlined_image, [largest_contour], -1, 255, 2)  # White color (thickness 2)

        return outlined_image

    # Helper function to perform inverse perspective transform
    def inverse_perspective_transform(self, outlined_image):
        contours, _ = cv2.findContours(outlined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None  # No contours found

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.01 * peri, True)  # 2% approximation

        if len(approx) != 4:
            # Not a quadrilateral; cannot perform perspective transform
            rospy.logwarn("Largest contour is not a quadrilateral. Skipping perspective transform.")
            return None

        # Order the points in consistent order: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2)
        rect = self.order_points(pts)

        # Compute the width and height of the new image
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Destination points for the perspective transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)

        # Apply the perspective transform
        warped = cv2.warpPerspective(outlined_image, M, (maxWidth, maxHeight))

        return warped

    # Helper function to order points
    def order_points(self, pts):
        # Initialize a list of coordinates that will be ordered
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and difference to find top-left and bottom-right
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]       # Top-left
        rect[2] = pts[np.argmax(s)]       # Bottom-right
        rect[1] = pts[np.argmin(diff)]    # Top-right
        rect[3] = pts[np.argmax(diff)]    # Bottom-left

        return rect

    def image_callback(self, msg):
        # Process mainfeed based on mainCombo selection (existing functionality)
        if self.mainCombo.currentText() == "Raw":
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

            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")

        # Process billboard based on billCombo selection
        bill_selection = self.billCombo.currentText()

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert the image to HSV color space
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for the target color (e.g., blue)
            lower_color = np.array([100, 120, 0])  
            upper_color = np.array([140, 255, 255]) 

            # Create a binary mask where the target color is white and the rest is black
            mask = cv2.inRange(hsv_image, lower_color, upper_color)

            # Apply morphological operations to remove noise and smooth the mask
            kernel = np.ones((1, 1), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Initialize variables
            processed_image = mask.copy()  # Start with the cleaned binary image
            quadrilateral_found = False  # Flag for billboard indicator

            if bill_selection == "Raw":
                # Display the cleaned binary image directly
                processed_image_display = mask

            elif bill_selection == "Contour":
                # Outline the largest quadrilateral contour on the cleaned binary image
                outlined_image = self.outline_largest_contour(mask)
                if outlined_image is not None:
                    processed_image_display = outlined_image
                else:
                    processed_image_display = mask

            elif bill_selection == "Homography":
                # Outline the largest quadrilateral contour on the cleaned binary image
                outlined_image = self.outline_largest_contour(mask)
                if outlined_image is not None:
                    # Attempt inverse perspective transform
                    warped_image = self.inverse_perspective_transform(outlined_image)
                    if warped_image is not None:
                        processed_image_display = warped_image
                        quadrilateral_found = True  # IPT successful
                    else:
                        processed_image_display = outlined_image
                else:
                    processed_image_display = mask

            else:
                rospy.logwarn(f"Unknown billCombo selection: {bill_selection}")
                processed_image_display = mask

            # Emit the signal with the processed image and bill selection
            self.image_update_signal.emit(processed_image_display, bill_selection)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    @QtCore.pyqtSlot(np.ndarray, str)
    def update_billboard(self, processed_image_display, bill_selection):
        # Convert processed image to QImage for display
        if len(processed_image_display.shape) == 2:
            # Grayscale image
            height, width = processed_image_display.shape
            bytes_per_line = width
            qt_image = QtGui.QImage(processed_image_display.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        else:
            # Color image (warped_image should be color if IPT was successful)
            processed_image_rgb = cv2.cvtColor(processed_image_display, cv2.COLOR_BGR2RGB)
            height, width, channel = processed_image_rgb.shape
            bytes_per_line = 3 * width
            qt_image = QtGui.QImage(processed_image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Scale the image to fit the billboard QLabel while maintaining aspect ratio
        scaled_image = qt_image.scaled(self.billboard.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        # Set the pixmap of the billboard QLabel
        self.billboard.setPixmap(QtGui.QPixmap.fromImage(scaled_image))

if __name__ == '__main__':
    # Initialize ROS node in the main thread if not already initialized
    if not rospy.core.is_initialized():
        rospy.init_node('controller_gui_node', anonymous=True)

    app = QtWidgets.QApplication(sys.argv)
    window = ControllerGUI()
    window.show()
    try:
        sys.exit(app.exec_())
    except rospy.ROSInterruptException:
        pass
