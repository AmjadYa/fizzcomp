#!/usr/bin/env python3

import sys
import os
import numpy as np
import rospkg
import rospy
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import datetime
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot
# from tensorflow.keras.models import load_model  # This can be removed if not used elsewhere
sys.path.append('/home/fizzer/fizzcomp/src/controller/src')
# print(os.path.abspath(__file__))
from teleport_functions import TeleportHandler
# Import prediction-related components from prediction_module.py
from prediction_module import load_cnn_model, PredictionThread, inverse_label_dict, IMAGE_WIDTH, IMAGE_HEIGHT
# Import the DataLogger class
from data_logger import DataLogger  
from bismillah_sequence import BismillahSequence  # <-- Added Import

class ControllerGUI(QtWidgets.QMainWindow):
    # Define a signal that carries the processed image and billCombo selection
    image_update_signal = pyqtSignal(np.ndarray, str)
    
    # Define a signal to send data to DataLogger
    data_signal = pyqtSignal(np.ndarray, float, float)
    
    # Define a signal to start the Bismillah sequence
    start_sequence_signal = pyqtSignal()

    def __init__(self):
        super(ControllerGUI, self).__init__()

        # Initialize TeleportHandler
        self.teleport_handler = TeleportHandler(model_name='B1')

        # Get the path to the 'controller' package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('controller')

        # Path to your CNN model file
        model_path = os.path.join(package_path, 'models', 'character_recognition_model.h5')

        # Load the CNN model using the function from prediction_module.py
        self.cnn_model = load_cnn_model(model_path)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Construct the full path to 'developer_tools.ui'
        ui_file = os.path.join(package_path, 'developer_tools.ui')

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

        # Set 'Homography' as the default option in billCombo
        bill_index = self.billCombo.findText("Homography")
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
        self.predict.clicked.connect(self.predict_image_function)  # New connection

        # Initialize set to keep track of pressed keys
        self.pressed_keys = set()

        # Movement flags controlled by buttons
        self.button_move_forward = False
        self.button_move_backward = False
        self.button_move_left = False
        self.button_move_right = False

        # Initialize movement commands
        self.current_linear_speed = 0.0
        self.current_angular_speed = 0.0

        # Start a timer to call publish_movement at regular intervals
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.publish_movement)
        self.timer.start(100)  # Every 100 ms (10 Hz)

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback)

        # Connect the image update signal to the update_billboard slot
        self.image_update_signal.connect(self.update_billboard)

        # Initialize the DataLogger
        self.data_logger = DataLogger()

        # Connect the data_signal to DataLogger's receive_data slot
        self.data_signal.connect(self.data_logger.receive_data)

        # Connect Record button and Record indicator
        self.monitor_manual_driving.clicked.connect(self.toggle_recording)
        self.is_recording = False  # Initial state
        self.update_record_indicator()

        # Ensure the window can accept focus and receive key events
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Default HSV bounds
        self.lower_color = np.array([0, 0, 99])
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

        # Connect TP Buttons to Teleport Functions using teleport_handler
        self.TP1.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP1'))
        self.TP2.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP2'))
        self.TP3.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP3'))
        self.TP4.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP4'))
        self.TP5.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP5'))
        self.TP6.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP6'))
        self.TP7.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP7'))
        self.TP8.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP8'))

        # Instantiate BismillahSequence
        self.bismillah_sequence = BismillahSequence(
            teleport_handler=self.teleport_handler
        )

        # Create a QThread and move BismillahSequence to it
        self.sequence_thread = QThread()
        self.bismillah_sequence.moveToThread(self.sequence_thread)

        # Connect signals from BismillahSequence to slots in ControllerGUI
        self.bismillah_sequence.set_linear.connect(self.set_linear_speed)
        self.bismillah_sequence.set_angular.connect(self.set_angular_speed)
        self.bismillah_sequence.stop_motion_signal.connect(self.stop_motion)
        self.bismillah_sequence.request_predict.connect(self.predict_image_function)
        self.bismillah_sequence.teleport_signal.connect(self.teleport_handler.teleport_to_pose)

        # Connect the start_sequence_signal to the BismillahSequence's run_sequence slot
        self.start_sequence_signal.connect(self.bismillah_sequence.run_sequence)

        # Start the sequence thread
        self.sequence_thread.start()

        # Connect the GOGOGO button to emit the start_sequence_signal
        self.GOGOGO.clicked.connect(self.start_sequence_signal.emit)

    def publish_movement(self):
        twist = Twist()
        twist.linear.x = self.current_linear_speed
        twist.angular.z = self.current_angular_speed
        self.pub_cmd_vel.publish(twist)
        # rospy.logdebug(f"Published Twist: linear.x={twist.linear.x}, angular.z={twist.angular.z}")

        # Emit data to DataLogger
        if hasattr(self, 'latest_image'):
            self.data_signal.emit(self.latest_image, self.current_linear_speed, self.current_angular_speed)
        else:
            # Handle case where latest_image is not yet available
            self.data_signal.emit(np.zeros((480, 640, 3), dtype=np.uint8), self.current_linear_speed, self.current_angular_speed)

    def set_linear_speed(self, speed):
        self.current_linear_speed = speed

    def set_angular_speed(self, speed):
        self.current_angular_speed = speed

    def stop_motion(self):
        self.current_linear_speed = 0.0
        self.current_angular_speed = 0.0

    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            self.data_logger.start_logging()
            self.is_recording = True
        else:
            # Stop recording
            self.data_logger.stop_logging()
            self.is_recording = False
        self.update_record_indicator()

    def update_record_indicator(self):
        if self.is_recording:
            self.record_indicator.setStyleSheet("QLabel { background-color: green; border-radius: 10px; }")
        else:
            self.record_indicator.setStyleSheet("QLabel { background-color: red; border-radius: 10px; }")

    def toggle_move_forward(self):
        self.button_move_forward = self.move_forward.isChecked()
        if self.button_move_forward:
            self.move_forward.setStyleSheet("background-color: green")
            self.current_linear_speed += 1.0
        else:
            self.move_forward.setStyleSheet("")
            self.current_linear_speed -= 1.0

    def toggle_move_backward(self):
        self.button_move_backward = self.move_backward.isChecked()
        if self.button_move_backward:
            self.move_backward.setStyleSheet("background-color: green")
            self.current_linear_speed -= 1.0
        else:
            self.move_backward.setStyleSheet("")
            self.current_linear_speed += 1.0

    def toggle_move_left(self):
        self.button_move_left = self.move_left.isChecked()
        if self.button_move_left:
            self.move_left.setStyleSheet("background-color: green")
            self.current_angular_speed += 1.0
        else:
            self.move_left.setStyleSheet("")
            self.current_angular_speed -= 1.0

    def toggle_move_right(self):
        self.button_move_right = self.move_right.isChecked()
        if self.button_move_right:
            self.move_right.setStyleSheet("background-color: green")
            self.current_angular_speed -= 1.0
        else:
            self.move_right.setStyleSheet("")
            self.current_angular_speed += 1.0

    # Keyboard event handlers
    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            if key == QtCore.Qt.Key_W:
                self.pressed_keys.add('W')
                self.current_linear_speed += 3.0
            elif key == QtCore.Qt.Key_S:
                self.pressed_keys.add('S')
                self.current_linear_speed -= 3.0
            elif key == QtCore.Qt.Key_A:
                self.pressed_keys.add('A')
                self.current_angular_speed += 2.0
            elif key == QtCore.Qt.Key_D:
                self.pressed_keys.add('D')
                self.current_angular_speed -= 2.0

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            if key == QtCore.Qt.Key_W:
                self.pressed_keys.discard('W')
                self.current_linear_speed -= 3.0
            elif key == QtCore.Qt.Key_S:
                self.pressed_keys.discard('S')
                self.current_linear_speed += 3.0
            elif key == QtCore.Qt.Key_A:
                self.pressed_keys.discard('A')
                self.current_angular_speed -= 2.0
            elif key == QtCore.Qt.Key_D:
                self.pressed_keys.discard('D')
                self.current_angular_speed += 2.0

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
        # Implement your auto drive functionality here
        pass

    # Helper functions for image processing (outline_largest_contour, inverse_perspective_transform, order_points)
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
        approx = cv2.approxPolyDP(largest_contour, 0.01 * peri, True)  # 1% approximation

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

                # Update the latest_image for logging
                self.latest_image = cv_image_rgb  # Store the raw RGB image

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
                kernel = np.ones((1, 1), np.uint8)
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


                # After setting the pixmap, also store the processed image
                self.latest_image = processed_image_display  # This could be a grayscale or color image

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

    def predict_image_function(self):
        """
        Handles the 'Predict' button click. Extracts the image from the billboard, segments it into letters,
        predicts each letter, and displays the results.
        """
        try:
            # Retrieve the current pixmap from the billboard
            pixmap = self.billboard.pixmap()
            if pixmap is None:
                rospy.logwarn("No image found on the billboard.")
                QtWidgets.QMessageBox.warning(self, "Warning", "No image found on the billboard.")
                return

            # Convert QPixmap to QImage
            qimage = pixmap.toImage()

            # Convert QImage to RGB888 format
            qimage = qimage.convertToFormat(QtGui.QImage.Format_RGB888)

            width = qimage.width()
            height = qimage.height()
            bytes_per_line = qimage.bytesPerLine()

            # Access the raw data
            ptr = qimage.bits()
            ptr.setsize(bytes_per_line * height)
            buffer = np.array(ptr).reshape(height, bytes_per_line)

            # Remove any padding bytes
            img = buffer[:, :width * 3].reshape(height, width, 3)

            # Start the prediction thread using the PredictionThread from prediction_module.py
            self.prediction_thread = PredictionThread(
                img, 
                self.cnn_model, 
                inverse_label_dict, 
                IMAGE_WIDTH, 
                IMAGE_HEIGHT
            )
            self.prediction_thread.prediction_complete.connect(self.on_prediction_complete)
            
            # Connect the prediction_complete signal to BismillahSequence's slot
            self.prediction_thread.prediction_complete.connect(self.bismillah_sequence.receive_prediction_result)
            
            self.prediction_thread.prediction_failed.connect(self.on_prediction_failed)
            self.prediction_thread.start()

        except Exception as e:
            rospy.logerr(f"Error during prediction: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during prediction: {e}")

    @QtCore.pyqtSlot(str)
    def on_prediction_complete(self, predicted_text):
        rospy.loginfo(f"Predicted Text: {predicted_text}")
        # Optionally, display the prediction result in the GUI
        # QtWidgets.QMessageBox.information(self, "Prediction Result", f"Predicted Text: {predicted_text}")

    @QtCore.pyqtSlot(str)
    def on_prediction_failed(self, error_message):
        rospy.logwarn(f"Prediction Failed: {error_message}")
        # Optionally, display a warning in the GUI
        # QtWidgets.QMessageBox.warning(self, "Prediction Failed", f"Prediction failed: {error_message}")

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

    def closeEvent(self, event):
        if self.is_recording:
            self.data_logger.stop_logging()
        # Gracefully stop the sequence thread
        if self.bismillah_sequence._is_running:
            self.bismillah_sequence._is_running = False
            self.sequence_thread.quit()
            self.sequence_thread.wait()
        event.accept()

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
