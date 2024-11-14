import cv2
import numpy as np

class RealTimeGastricSiteRecognition:
    def __init__(self, model, image_size, label_shift):
        """
        Initializes the RealTimeGastricSiteRecognition object for recognizing digits in video frames.

        Parameters:
        - **model (object)**: The digit recognition model used for identifying digits.

        Returns:
        - **None**: This method initializes the attributes of the object but does not return any values.
        """
        self.model = model
        self.image_size = image_size
        self.seen_digits = set()  # Use set for faster lookups
        self.update_digits = False
        self.label_shift = label_shift

    def preprocess_frame(self, frame):
        """Resizes and converts the frame to grayscale for model prediction."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (self.image_size[0], self.image_size[1]))
        return resized_frame.reshape(1, self.image_size[0], self.image_size[1], self.image_size[2]).astype('float32') / 255

    def predict_digit(self, processed_frame):
        """Runs the model prediction and returns the predicted class."""
        predictions = self.model.predict(processed_frame)

        print(predictions)
        max_prediction = np.max(predictions)
        predicted_class = np.argmax(predictions)
        if max_prediction < 0.85:
            return -1
        if self.label_shift:
            return (predicted_class -1) #Label shift so return original prediction - 1!
        else:
            return (predicted_class) #No Label shift so return original prediction!

    def process_frame(self, frame):
        """Processes a single frame and updates the seen digits list if required."""
        processed_frame = self.preprocess_frame(frame)
        predicted_class = self.predict_digit(processed_frame)

        # Update seen digits if the process is running and the digit is valid
        if self.update_digits and predicted_class != -1 and predicted_class not in self.seen_digits:
            self.seen_digits.add(predicted_class)

        return predicted_class