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
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelState
from tensorflow.keras.models import load_model

class ControllerGUI(QtWidgets.QMainWindow):
    # Define a signal that carries the processed image and billCombo selection
    image_update_signal = pyqtSignal(np.ndarray, str)

    def __init__(self):
        super(ControllerGUI, self).__init__()

        # Initialize ROS node
        rospy.init_node('controller_gui_node', anonymous=True)

        # Get the path to the 'controller' package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('controller')

        # Path to your CNN model file
        model_path = os.path.join(package_path, 'models', 'character_recognition_model.h5')

        # Try to load the model
        try:
            self.cnn_model = load_model(model_path, compile=False)
            rospy.loginfo(f"Successfully loaded CNN model from {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load CNN model: {e}")
            sys.exit(1)

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

        # Default HSV bounds
        self.lower_color = np.array([0, 0, 174])
        self.upper_color = np.array([179, 91, 255])

        # Set default slider values for lower bounds
        self.hSlider.setMinimum(0)
        self.hSlider.setMaximum(179)
        self.hSlider.setValue(self.lower_color[0])
        self.sSlider.setMinimum(0)
        self.sSlider.setMaximum(255)
        self.sSlider.setValue(self.lower_color[1])
        self.vSlider.setMinimum(0)
        self.vSlider.setMaximum(255)
        self.vSlider.setValue(self.lower_color[2])

        # Set default slider values for upper bounds
        self.hSlider_2.setMinimum(0)
        self.hSlider_2.setMaximum(179)
        self.hSlider_2.setValue(self.upper_color[0])
        self.sSlider_2.setMinimum(0)
        self.sSlider_2.setMaximum(255)
        self.sSlider_2.setValue(self.upper_color[1])
        self.vSlider_2.setMinimum(0)
        self.vSlider_2.setMaximum(255)
        self.vSlider_2.setValue(self.upper_color[2])

        # Update labels with default slider values
        self.hText.setText(str(self.hSlider.value()))
        self.sText.setText(str(self.sSlider.value()))
        self.vText.setText(str(self.vSlider.value()))
        self.hText_2.setText(str(self.hSlider_2.value()))
        self.sText_2.setText(str(self.sSlider_2.value()))
        self.vText_2.setText(str(self.vSlider_2.value()))

        # Connect sliders to their respective update functions
        self.hSlider.valueChanged.connect(self.update_hText)
        self.sSlider.valueChanged.connect(self.update_sText)
        self.vSlider.valueChanged.connect(self.update_vText)
        self.hSlider_2.valueChanged.connect(self.update_hText_2)
        self.sSlider_2.valueChanged.connect(self.update_sText_2)
        self.vSlider_2.valueChanged.connect(self.update_vText_2)

        self.teleport_positions = {
            'TP1': {
                'position': {'x': 5.49942880783774, 'y': 2.504030700996579, 'z': 0.04000039797012949},
                'orientation': {'x': 7.377270669093508e-07, 'y': -4.3205541389025773e-07, 'z': -0.7090994033738673, 'w': 0.7051085279119056}
            },
            'TP2': {
                'position': {'x': 5.3474887443153785, 'y': -0.9349569584188745, 'z': 0.04000037723629943},
                'orientation': {'x': 1.7321540120358244e-06, 'y': 8.610931097589084e-07, 'z': -0.814923731390067, 'w': 0.5795682117003553}
            },
            'TP3': {
                'position': {'x': 4.31305060031579, 'y': -1.3878525178704242, 'z': 0.04000050138909643},
                'orientation': {'x': 2.8548229122584364e-07, 'y': 5.904484287748461e-08, 'z': -0.9992515317144463, 'w': -0.038683024264505914}
            },
            'TP4': {
                'position': {'x': 0.6491008198665422, 'y': -0.9207976055506509, 'z': 0.04000053773996677},
                'orientation': {'x': 5.673153474413703e-07, 'y': 1.1238297852020603e-07, 'z': -0.6684896009764731, 'w': -0.7437214891247805}
            },
            'TP5': {
                'position': {'x': 0.6689328884773375, 'y': 2.01851026363387, 'z': 0.04000094070292316},
                'orientation': {'x': -2.860153819036568e-06, 'y': 3.005876131366921e-06, 'z': 0.7085922122246099, 'w': -0.7056182230905165}
            },
            'TP6': {
                'position': {'x': -3.0218217082015695, 'y': 1.5572375439630923, 'z': 0.03997891270315111},
                'orientation': {'x': -4.068919485243806e-05, 'y': -9.436718605393764e-05, 'z': 0.9984824087289902, 'w': 0.05507148897549148}
            },
            'TP7': {
                'position': {'x': -4.301637175421005, 'y': -2.312349652507272, 'z': 0.03998563711874584},
                'orientation': {'x': -6.732302218612708e-06, 'y': 0.00012692216855302477, 'z': -0.05064239775258873, 'w': -0.9987168424510062}
            },
            'TP8': {
                'position': {'x': -1.2089846301463472, 'y': -1.186840844647, 'z': 1.8503400563324273},
                'orientation': {'x': -0.0057606116672731245, 'y': -0.013744051998107805, 'z': -0.0008294302102146539, 'w': -0.9998886080126218}
            }
        }

        # ----- New Section: Set Up Teleport Service Proxy -----
        rospy.loginfo("Waiting for /gazebo/set_model_state service...")
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.loginfo("/gazebo/set_model_state service is available.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service initialization failed: {e}")
            sys.exit(1)
        # ----- End of New Section -----

        # ----- New Section: Connect TP Buttons to Teleport Functions -----
        self.TP1.clicked.connect(lambda: self.teleport_to_position('TP1'))
        self.TP2.clicked.connect(lambda: self.teleport_to_position('TP2'))
        self.TP3.clicked.connect(lambda: self.teleport_to_position('TP3'))
        self.TP4.clicked.connect(lambda: self.teleport_to_position('TP4'))
        self.TP5.clicked.connect(lambda: self.teleport_to_position('TP5'))
        self.TP6.clicked.connect(lambda: self.teleport_to_position('TP6'))
        self.TP7.clicked.connect(lambda: self.teleport_to_position('TP7'))
        self.TP8.clicked.connect(lambda: self.teleport_to_position('TP8'))
        # ----- End of New Section -----

    # ----- New Section: Teleport Function -----
    def teleport_to_position(self, tp_name):
        """
        Teleports the robot to the specified TP position.

        :param tp_name: String name of the TP button (e.g., 'TP1', 'TP2', ...)
        """
        if tp_name not in self.teleport_positions:
            rospy.logerr(f"Teleport position '{tp_name}' not defined.")
            return

        position = self.teleport_positions[tp_name]['position']
        orientation = self.teleport_positions[tp_name]['orientation']

        # Create a ModelState message
        model_state = ModelState()
        model_state.model_name = 'B1'  # Ensure this matches your robot's model name in Gazebo
        model_state.pose.position.x = position['x']
        model_state.pose.position.y = position['y']
        model_state.pose.position.z = position['z']
        model_state.pose.orientation.x = orientation['x']
        model_state.pose.orientation.y = orientation['y']
        model_state.pose.orientation.z = orientation['z']
        model_state.pose.orientation.w = orientation['w']
        model_state.reference_frame = 'world'  # Relative to the 'world' frame

        # Create the service request
        set_state_request = SetModelStateRequest()
        set_state_request.model_state = model_state

        try:
            # Call the service to set the model state
            response = self.set_model_state_service(set_state_request)
            if response.success:
                rospy.loginfo(f"Successfully teleported to {tp_name}.")
            else:
                rospy.logerr(f"Failed to teleport to {tp_name}: {response.status_message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
    # ----- End of New Section -----

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
        # Process mainfeed based on mainCombo selection
        main_selection = self.mainCombo.currentText()

        if main_selection == "Raw":
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

        elif main_selection == "HSV":
            try:
                # Retrieve current HSV bounds from sliders
                lower_h = self.hSlider.value()
                lower_s = self.sSlider.value()
                lower_v = self.vSlider.value()
                upper_h = self.hSlider_2.value()
                upper_s = self.sSlider_2.value()
                upper_v = self.vSlider_2.value()

                # Update lower and upper color arrays
                lower_color = np.array([lower_h, lower_s, lower_v])
                upper_color = np.array([upper_h, upper_s, upper_v])

                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                # Convert the image to HSV color space
                hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

                # Create a binary mask where the target color is white and the rest is black
                mask = cv2.inRange(hsv_image, lower_color, upper_color)

                # Apply morphological operations to remove noise and smooth the mask
                kernel = np.ones((1, 10), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

                # Display the processed mask
                processed_image_display = mask

                # Convert processed image to QImage for display
                height, width = processed_image_display.shape
                bytes_per_line = width
                qt_image = QtGui.QImage(processed_image_display.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)

                # Scale the image to fit the mainfeed QLabel while maintaining aspect ratio
                scaled_image = qt_image.scaled(self.mainfeed.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

                # Set the pixmap of the mainfeed QLabel
                self.mainfeed.setPixmap(QtGui.QPixmap.fromImage(scaled_image))

            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")

        else:
            rospy.logwarn(f"Unknown mainCombo selection: {main_selection}")
            # Optionally, handle other cases or default behavior

        # Process billboard based on billCombo selection
        bill_selection = self.billCombo.currentText()

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert the image to HSV color space
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for the target color (e.g., blue)
            lower_color_bill = np.array([100, 120, 0])  
            upper_color_bill = np.array([140, 255, 255]) 

            # Create a binary mask where the target color is white and the rest is black
            mask = cv2.inRange(hsv_image, lower_color_bill, upper_color_bill)

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

    # ----- Added Section: Slider Update Functions -----
    def update_hText(self, value):
        self.hText.setText(str(value))

    def update_sText(self, value):
        self.sText.setText(str(value))

    def update_vText(self, value):
        self.vText.setText(str(value))

    def update_hText_2(self, value):
        self.hText_2.setText(str(value))

    def update_sText_2(self, value):
        self.sText_2.setText(str(value))

    def update_vText_2(self, value):
        self.vText_2.setText(str(value))
    # ----- End of Added Section -----

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
