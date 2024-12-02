#!/usr/bin/env python3

import sys
import os
from matplotlib import pyplot as plt
import rospkg
import rospy
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import datetime
from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelState
from tensorflow.keras.models import load_model
import math
from teleport_functions import TeleportHandler

# Define image dimensions
IMAGE_WIDTH, IMAGE_HEIGHT = 34, 55  # Adjust as needed based on your data

# Define label dictionary (example)
label_dict = {
    'A': 0, 
    'B': 1, 
    'C': 2, 
    'D': 3, 
    'E': 4, 
    'F': 5, 
    'G': 6, 
    'H': 7, 
    'I': 8, 
    'J': 9, 
    'K': 10, 
    'L': 11, 
    'M': 12, 
    'N': 13, 
    'O': 14, 
    'P': 15, 
    'Q': 16, 
    'R': 17, 
    'S': 18, 
    'T': 19, 
    'U': 20, 
    'V': 21, 
    'W': 22, 
    'X': 23, 
    'Y': 24, 
    'Z': 25, 
    '0': 26, 
    '1': 27, 
    '2': 28, 
    '3': 29, 
    '4': 30, 
    '5': 31, 
    '6': 32, 
    '7': 33, 
    '8': 34, 
    '9': 35, 
    '': 36
}

def extract_letters_from_image(img, base_width=34, base_height=55, tolerance=5, extend_width=36, extend_height=73, space_threshold=30):
    """
    Extract letters from an image represented as a NumPy array.

    :param img: Input image as a NumPy array (BGR format).
    :param base_width: Base width for letters.
    :param base_height: Base height for letters.
    :param tolerance: Tolerance for width and height.
    :param extend_width: Width to extend for smaller bounding boxes.
    :param extend_height: Height to extend for smaller bounding boxes.
    :param space_threshold: Threshold to detect spaces between letters.
    :return: List of extracted letter images as NumPy arrays.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Work with the lower half of the image
    height = gray.shape[0]
    lower_half = gray[height // 2:, :]

    # Threshold the lower half to binary for contour detection
    _, thresh = cv2.threshold(lower_half, 90, 255, cv2.THRESH_BINARY)

    # Find contours in the lower half
    contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_lower = contours_info[-2] if len(contours_info) >= 2 else []
    if not contours_lower:
        print("No letter contours found in the lower half of the image.")
        return []

    # Get bounding boxes and filter out full-width boxes
    bounding_boxes = [cv2.boundingRect(c) for c in contours_lower]
    bounding_boxes = [b for b in bounding_boxes if b[2] < lower_half.shape[1]]  # Exclude full-width boxes
    bounding_boxes.sort(key=lambda b: b[0])  # Sort by x-coordinate

    letters, positions = [], []
    for x, y, w, h in bounding_boxes:
        # Process valid bounding boxes
        if (base_width - tolerance) <= w <= (base_width + tolerance) and (base_height - tolerance) <= h <= (base_height + tolerance):
            letters.append(lower_half[y:y+h, x:x+w])
            positions.append((x, x + w))
        elif w > (base_width + tolerance):  # Handle oversized bounding boxes
            num_letters = math.ceil(w / (base_width + tolerance))
            divided_width = w / num_letters
            for i in range(num_letters):
                x_start = int(x + i * divided_width)
                x_end = int(x_start + divided_width)
                if i == num_letters - 1:  # Ensure last segment doesn't exceed the box
                    x_end = x + w
                crop = lower_half[y:y+h, x_start:x_end]
                letters.append(cv2.resize(crop, (base_width, base_height)))
                positions.append((x_start, x_end))
        else:  # Handle smaller-than-expected bounding boxes
            x_end = min(x + extend_width, lower_half.shape[1])
            y_end = min(y + extend_height, lower_half.shape[0])
            crop = lower_half[y:y_end, x:x_end]
            letters.append(cv2.resize(crop, (base_width, base_height)))
            positions.append((x, x_end))

    # Add spaces between letters where needed
    final_letters, prev_x_end = [], 0
    for idx, letter in enumerate(letters):
        x_start, x_end = positions[idx]
        if idx > 0 and (x_start - prev_x_end) > space_threshold:
            final_letters.append(np.zeros((base_height, base_width), dtype=np.uint8))  # Add a blank space
        final_letters.append(letter)
        prev_x_end = x_end

    # Optional: Display results
    # Uncomment if you want to visualize the letters
    num_letters = len(final_letters)
    cols = min(10, num_letters)
    rows = math.ceil(num_letters / cols)

    plt.figure(figsize=(cols * 2, rows * 2))
    for idx, letter in enumerate(final_letters):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(letter, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Total letters and spaces extracted: {len(final_letters)}")
    return final_letters

class PredictionThread(QThread):
    prediction_complete = pyqtSignal(str)
    prediction_failed = pyqtSignal(str)

    def __init__(self, img, model, inverse_label_dict, image_width, image_height, parent=None):
        super(PredictionThread, self).__init__(parent)
        self.img = img
        self.model = model
        self.inverse_label_dict = inverse_label_dict
        self.image_width = image_width
        self.image_height = image_height

    def run(self):
        try:
            letters = extract_letters_from_image(self.img)
            if not letters:
                self.prediction_failed.emit("No letters were extracted from the image.")
                return

            predictions = []
            for letter_img in letters:
                if self.model is None:
                    self.prediction_failed.emit("CNN model is not loaded.")
                    return
                # Predict each letter
                if len(letter_img.shape) == 2:
                    letter_img = cv2.cvtColor(letter_img, cv2.COLOR_GRAY2BGR)
                img_resized = cv2.resize(letter_img, (self.image_width, self.image_height))
                img_normalized = img_resized / 255.0
                img_expanded = np.expand_dims(img_normalized, axis=0)
                predictions_array = self.model.predict(img_expanded)
                predicted_index = np.argmax(predictions_array, axis=1)[0]
                predicted_label = self.inverse_label_dict.get(predicted_index, "?")
                predictions.append(predicted_label)

            predicted_text = ''.join(predictions)
            self.prediction_complete.emit(predicted_text)

        except Exception as e:
            self.prediction_failed.emit(str(e))

class ControllerGUI(QtWidgets.QMainWindow):
    # Define a signal that carries the processed image and billCombo selection
    image_update_signal = pyqtSignal(np.ndarray, str)

    def __init__(self):
        super(ControllerGUI, self).__init__()

        # Initialize ROS node
        rospy.init_node('controller_gui_node', anonymous=True)

        # Initialize TeleportHandler
        self.teleport_handler = TeleportHandler(model_name='B1')

        # Get the path to the 'controller' package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('controller')

        # Path to your CNN model file
        model_path = os.path.join(package_path, 'models', 'character_recognition_model.h5')

        # Define your label dictionary (example)
        self.label_dict = label_dict

        # Create inverse label dictionary for mapping indices to labels
        self.inverse_label_dict = {v: k for k, v in self.label_dict.items()}

        # Load the CNN model once during initialization
        self.cnn_model = self.load_cnn_model(model_path)

        # Initialize CvBridge
        self.bridge = CvBridge()

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
        self.predict.clicked.connect(self.predict_image_function)  # New connection

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

        # ----- New Section: Connect TP Buttons to Teleport Functions -----
        # Connect TP Buttons to Teleport Functions using teleport_handler
        self.TP1.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP1'))
        self.TP2.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP2'))
        self.TP3.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP3'))
        self.TP4.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP4'))
        self.TP5.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP5'))
        self.TP6.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP6'))
        self.TP7.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP7'))
        self.TP8.clicked.connect(lambda: self.teleport_handler.teleport_to_position('TP8'))

        # ----- End of New Section -----

    def teleport_to_position(self, tp_name):
        pass

    def load_cnn_model(self, model_path):
        """
        Load the CNN model from the specified path.

        :param model_path: Path to the CNN model file.
        :return: Loaded Keras model.
        """
        try:
            model = load_model(model_path, compile=False)
            rospy.loginfo(f"Successfully loaded CNN model from {model_path}")
            return model
        except Exception as e:
            rospy.logerr(f"Failed to load CNN model: {e}")
            sys.exit(1)

    def toggle_move_forward(self):
        self.button_move_forward = self.move_forward.isChecked()
        if self.button_move_forward:
            self.move_forward.setStyleSheet("background-color: green")
        else:
            self.move_forward.setStyleSheet("")

    def toggle_move_backward(self):
        self.button_move_backward = self.move_backward.isChecked()
        if self.button_move_backward:
            self.move_backward.setStyleSheet("background-color: green")
        else:
            self.move_backward.setStyleSheet("")

    def toggle_move_left(self):
        self.button_move_left = self.move_left.isChecked()
        if self.button_move_left:
            self.move_left.setStyleSheet("background-color: green")
        else:
            self.move_left.setStyleSheet("")

    def toggle_move_right(self):
        self.button_move_right = self.move_right.isChecked()
        if self.button_move_right:
            self.move_right.setStyleSheet("background-color: green")
        else:
            self.move_right.setStyleSheet("")

    # Keyboard event handlers
    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            if key in [QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D]:
                self.pressed_keys.add(key)

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            if key in [QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D]:
                self.pressed_keys.discard(key)

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
        # Implement your auto drive functionality here
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
            qimage = qimage.convertToFormat(QtGui.QImage.Format_RGB888)  # Keep RGB

            width = qimage.width()
            height = qimage.height()

            # Retrieve image data as bytes
            buffer = qimage.bits().asstring(qimage.byteCount())

            # Convert bytes to NumPy array
            img = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)  # RGB format

            # Start the prediction thread
            self.prediction_thread = PredictionThread(
                img, 
                self.cnn_model, 
                self.inverse_label_dict, 
                IMAGE_WIDTH, 
                IMAGE_HEIGHT
            )
            self.prediction_thread.prediction_complete.connect(self.on_prediction_complete)
            self.prediction_thread.prediction_failed.connect(self.on_prediction_failed)
            self.prediction_thread.start()

        except Exception as e:
            rospy.logerr(f"Error during prediction: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during prediction: {e}")

    @QtCore.pyqtSlot(str)
    def on_prediction_complete(self, predicted_text):
        rospy.loginfo(f"Predicted Text: {predicted_text}")
        QtWidgets.QMessageBox.information(self, "Prediction Result", f"Predicted Text: {predicted_text}")

    @QtCore.pyqtSlot(str)
    def on_prediction_failed(self, error_message):
        rospy.logwarn(f"Prediction Failed: {error_message}")
        QtWidgets.QMessageBox.warning(self, "Prediction Failed", f"Prediction failed: {error_message}")

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
