import cv2
import numpy as np

class RealTimeInsideOutsideRecognition:
    def __init__(self, model, image_size):
        """
        Initializes the RealTimeInsideOutsideRecognition object for recognizing if the endoscope is inside of outside the GI Tract in video frames.
        """
        self.model = model
        self.image_size = image_size

    def preprocess_frame(self, frame):
        """Resizes and converts the frame to grayscale for model prediction."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resized_frame = cv2.resize(rgb_frame, (self.image_size[0], self.image_size[1]))

        return resized_frame.reshape(1, self.image_size[0], self.image_size[1], self.image_size[2]).astype('float32') / 255

    def predict_digit(self, processed_frame):
        """Runs the model prediction and returns the predicted class."""
        predictions = self.model.predict(processed_frame)
        
        max_prediction = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        if max_prediction < 0.95:
            return -1
        return predicted_class

    def process_frame(self, frame):
        """Processes a single frame and updates the seen digits list if required."""
        processed_frame = self.preprocess_frame(frame)
        predicted_class = self.predict_digit(processed_frame)

        return predicted_class