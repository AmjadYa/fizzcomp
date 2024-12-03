# prediction_module.py

import math
import sys
import os
from PyQt5.QtCore import pyqtSignal, QThread
from tensorflow.keras.models import load_model
import numpy as np
import cv2

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

# Create inverse label dictionary for mapping indices to labels
inverse_label_dict = {v: k for k, v in label_dict.items()}

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
    #num_letters = len(final_letters)
    #cols = min(10, num_letters)
    #rows = math.ceil(num_letters / cols)

    #plt.figure(figsize=(cols * 2, rows * 2))
    #for idx, letter in enumerate(final_letters):
    #    plt.subplot(rows, cols, idx + 1)
    #    plt.imshow(letter, cmap='gray')
    #    plt.axis('off')
    #plt.tight_layout()
    #plt.show()

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

def load_cnn_model(model_path):
    """
    Load the CNN model from the specified path.

    :param model_path: Path to the CNN model file.
    :return: Loaded Keras model.
    """
    try:
        model = load_model(model_path, compile=False)
        print(f"Successfully loaded CNN model from {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load CNN model: {e}")
        sys.exit(1)
