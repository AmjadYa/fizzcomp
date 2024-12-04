# twist_prediction_module.py

import sys
import os
from PyQt5.QtCore import pyqtSignal, QThread
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from geometry_msgs.msg import Twist

# Define image dimensions
IMAGE_WIDTH, IMAGE_HEIGHT = 300, 300  # Adjust based on your CNN input

# Define inverse label dictionaries
linear_label_dict = {
    0: -3.0,
    1: 0.0,
    2: 3.0
}

angular_label_dict = {
    0: -2.0,
    1: 0.0,
    2: 2.0
}

def load_twist_model(model_path):
    """
    Load the Twist prediction CNN model from the specified path.

    :param model_path: Path to the CNN model file.
    :return: Loaded Keras model.
    """
    try:
        model = load_model(model_path, compile=False)
        print(f"Successfully loaded Twist CNN model from {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load Twist CNN model: {e}")
        sys.exit(1)

class TwistPredictionThread(QThread):
    twist_signal = pyqtSignal(Twist)

    def __init__(self, model, parent=None):
        super(TwistPredictionThread, self).__init__(parent)
        self.model = model
        self.latest_image = None
        self._running = True

    def run(self):
        while self._running:
            if self.latest_image is not None:
                try:
                    # Preprocess the image
                    img = self.preprocess_image(self.latest_image)

                    # Predict the twists
                    predictions = self.model.predict(img)
                    
                    # Ensure predictions have two outputs
                    if isinstance(predictions, list) and len(predictions) == 2:
                        linear_pred = np.argmax(predictions[0], axis=1)[0]
                        angular_pred = np.argmax(predictions[1], axis=1)[0]
                    else:
                        print("Model predictions do not have two outputs as expected.")
                        continue

                    # Map predictions to actual values
                    linear_x = linear_label_dict.get(linear_pred, 0.0)
                    angular_z = angular_label_dict.get(angular_pred, 0.0)

                    # Create Twist message
                    twist_msg = Twist()
                    twist_msg.linear.x = linear_x
                    twist_msg.angular.z = angular_z

                    # Emit the Twist message
                    self.twist_signal.emit(twist_msg)

                except Exception as e:
                    print(f"Error in TwistPredictionThread: {e}")

            self.msleep(100)  # Sleep for 100 ms

    def preprocess_image(self, img):
        """
        Preprocess the image for CNN prediction.

        :param img: Input image as a NumPy array.
        :return: Preprocessed image ready for prediction.
        """
        if len(img.shape) == 2:
            # If grayscale, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            # If single channel, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # If RGBA, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            # Assume already BGR
            pass

        # Resize the image to match the CNN input
        img_resized = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Normalize pixel values to [0, 1]
        img_normalized = img_resized.astype('float32') / 255.0

        # Expand dimensions to match model input
        img_expanded = np.expand_dims(img_normalized, axis=0)

        return img_expanded

    def receive_image(self, img):
        """
        Receive the latest image for prediction.

        :param img: Latest image as a NumPy array.
        """
        self.latest_image = img

    def stop(self):
        """
        Stop the prediction thread.
        """
        self._running = False
