import cv2
import numpy as np

class RealTimeDigitRecognition:
    def __init__(self, model, image_size, label_shift, downscale_factor=0.5):
        """
        Initializes the RealTimeDigitRecognition object for recognizing digits in video frames.

        Parameters:
        - **model (object)**: The digit recognition model used for identifying digits.
        - **frame_skip (int, optional)**: Number of frames to skip between processing, default is 3.
        - **downscale_factor (float, optional)**: Factor to downscale the video frames, default is 0.5.

        Attributes:
        - **model**: The digit recognition model.
        - **frame_skip**: Frames to skip between processing.
        - **downscale_factor**: Factor for downscaling frames.
        - **seen_digits (set)**: Set to store unique digits that have been recognized.
        - **frame_count (int)**: Counter for the number of frames processed.
        - **update_digits (bool)**: Flag to control whether digits should be updated.

        Returns:
        - **None**: This method initializes the attributes of the object but does not return any values.
        """
        self.model = model
        self.image_size = image_size
        self.downscale_factor = downscale_factor
        self.seen_digits = set()  # Use set for faster lookups
        self.frame_count = 0
        self.update_digits = False
        self.label_shift = label_shift

    def preprocess_frame(self, frame):
        """Resizes and converts the frame to grayscale for model prediction."""
        # Original frame
        # cv2.imshow('Original Frame', frame)

        #frame = cv2.resize(frame, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor)
        #cv2.imshow('Resized Frame', frame)
        
        #cv2.imshow('Grayscale Frame', gray_frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # USE THIS TO SHOW THE FRAME WITH CORRECT COLOUR
        # cv2.imshow('NORMAL', bgr_frame)

        resized_frame = cv2.resize(rgb_frame, (self.image_size[0], self.image_size[1]))
        # cv2.imshow('BLUE', resized_frame)
        # cv2.imshow('NORMAL', bgr_frame)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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