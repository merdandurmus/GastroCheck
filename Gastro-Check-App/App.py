# The provided class `Application` sets up a real-time digit recognition and motion tracking
# application with a user interface in Python, utilizing a trained CNN model for digit recognition and
# an NDITracker for motion tracking.
import os
import csv
import time
import threading
import math
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import Frame, messagebox
from tkinter import ttk

from tensorflow.keras.models import load_model # type: ignore
from sksurgerynditracker.nditracker import NDITracker



class RealTimeDigitRecognition:
    def __init__(self, model, image_size, labelshift, downscale_factor=0.5):
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
        self.labelshift = labelshift

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
        # Convert the processed_frame back to a displayable format for OpenCV
        #display_frame = (processed_frame[0] * 255).astype(np.uint8)  # Remove batch dimension and scale back to 0-255

        """Runs the model prediction and returns the predicted class."""
        predictions = self.model.predict(processed_frame)

        # cv2.imshow('BLUE', display_frame)
        # bgr_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow('NORMAL', bgr_frame)
        # cv2.waitKey(1)  # Small delay for display

        print(predictions)
        max_prediction = np.max(predictions)
        predicted_class = np.argmax(predictions)
        if max_prediction < 0.85:
            return -1
        if self.labelshift:
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

    def display_frame(self, frame, predicted_class, blinking_dot):
        """Displays the predicted digit and a blinking dot if the process is running."""
        if predicted_class != -1:
            cv2.putText(frame, f"Predicted Digit: {predicted_class}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if blinking_dot:
            cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)  # Red dot at the top-left corner


class VideoFeed:
    def __init__(self, window, video_frame, digit_recognizer, array_data_label, digit_show_label, gi_label, num_classes):
        """
        Initializes the VideoFeed object for capturing and displaying video frames.

        Parameters:
        - **window (Tk)**: The root Tkinter window where the video feed will be displayed.
        - **video_frame (Label)**: The label widget where video frames will be shown.
        - **digit_recognizer (RealTimeDigitRecognition)**: The digit recognition system to process video frames.
        - **array_data_label (Label)**: The label widget for displaying recognized digits.

        Initializes the video capture, sets up control flags, and starts the video feed.

        Returns:
        - **None**: This method sets up the video capture and related settings but does not return any values.
        """
        self.window = window
        self.video_frame = video_frame
        self.gi_label = gi_label
        self.cap = cv2.VideoCapture(0) #1 FOR VIDEO CAPTURE
        self.running = True
        self.digit_recognizer = digit_recognizer
        self.array_data_label = array_data_label
        self.digit_show_label = digit_show_label
        self.blinking_dot = False  # Control blinking red dot for display
        self.frame_update_delay = 100  # Delay in ms for video frame updates
        self.num_classes = num_classes

        self.start_video_feed()

    def start_video_feed(self):
        """Starts the video feed and handles real-time digit recognition."""
        if self.running:
            ret, frame = self.read_frame()
            if ret:
                predicted_class = self.process_frame_and_recognize_digit(frame)
                self.display_frame(frame, predicted_class)
                self.update_detecting_digits_display(predicted_class)

                # Update seen digits display if procedure is running
                if self.digit_recognizer.update_digits:
                    self.update_seen_digits_display()
                
            # Schedule the next video frame update
            self.window.after(self.frame_update_delay, self.start_video_feed)

    def read_frame(self):
        """Reads a frame from the video feed."""
        return self.cap.read()

    def process_frame_and_recognize_digit(self, frame):
        """Processes the frame for digit recognition and returns the predicted class."""
        predicted_class = self.digit_recognizer.process_frame(frame)
        return predicted_class

    def display_frame(self, frame, predicted_class):
        """Displays the processed frame in the UI with the predicted class and optional blinking dot."""
        self.digit_recognizer.display_frame(frame, predicted_class, self.blinking_dot)
        frame_rgb = self.convert_frame_to_rgb(frame)
        self.update_tkinter_image(frame_rgb)

    def convert_frame_to_rgb(self, frame):
        """Converts the frame to RGB format."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def update_tkinter_image(self, frame_rgb):
        """Converts a frame to Tkinter-compatible format and updates the video display."""
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.config(image=imgtk)

    def update_seen_digits_display(self):
        """Updates the label that displays the seen digits."""
        seen_digits_text = ", ".join(map(str, self.digit_recognizer.seen_digits))
        self.array_data_label.config(text=seen_digits_text)
        
        # CHANGE IMAGE
        image_folder = "GI-Tract-Images/"
        
        # Initialize the background with the default image
        background = Image.open(os.path.join(image_folder, "0.png")).convert("RGBA")
        
        if seen_digits_text != "":
            numbers_list = sorted([int(num.strip()) for num in seen_digits_text.split(',')])
            
            # Check if exactly 4 digits are recognized
            if len(numbers_list) == self.num_classes:
                image_name = "SeenFULL.png"
                background = Image.open(os.path.join(image_folder, image_name)).convert("RGBA")
            else:
                # Loop through recognized digits and overlay images
                for n in numbers_list:
                    image_path = os.path.join(image_folder, f"{n}.png")
                    overlay = Image.open(image_path).convert("RGBA")

                    # Resize overlay image to match background if necessary
                    overlay = overlay.resize(background.size)

                    # Set the transparency level (0 - fully transparent, 255 - fully opaque)
                    overlay.putalpha(128)  # For 50% transparency

                    # Stack the images on top of each other
                    background = Image.alpha_composite(background, overlay)

        # Resize and update the label with the new image
        self.gi_image = background.resize((400, 600), Image.Resampling.LANCZOS)
        gi_image_tk = ImageTk.PhotoImage(self.gi_image)
        
        # Keep a reference to avoid garbage collection
        self.gi_image.imgtk = gi_image_tk
        self.gi_label.config(image=gi_image_tk)

    def update_detecting_digits_display(self, predicted_class):
        """Updates the label that displays the seen digits."""
        self.digit_show_label.config(text=predicted_class)

    def stop(self):
        """Stops the video feed and releases the camera."""
        self.running = False
        self.cap.release()

class Application:
    def __init__(self, root, model, tracker, image_size, labelshift, num_classes):
        """
        Initializes the application by setting up the main window and configuring necessary components.

        This method performs the following actions:

        1. **Sets Up the Main Window**:
        - Sets the title of the main window to "Gastro-Check".
        - Configures the background color of the main window to light sky blue (`bg="LightSkyBlue1"`).

        2. **Initializes the Digit Recognizer**:
        - Creates an instance of `RealTimeDigitRecognition` with the provided `model`, which is used for recognizing digits in real time.

        3. **Sets Up the User Interface**:
        - Calls `self.setup_ui()` to initialize and arrange the various UI components, including frames and labels.

        4. **Initializes the Video Feed**:
        - Creates an instance of `VideoFeed` with parameters including `self.root`, `self.video_frame`, `self.digit_recognizer`, and `self.array_data_label` to handle video processing and digit recognition.

        5. **Initializes Other Attributes**:
        - Sets `self.start_time` to `None`, indicating that the procedure has not started yet.
        - Sets `self.running_timer` to `False`, indicating that the timer is not running.
        - Sets `self.is_procedure_running` to `False`, indicating that no procedure is currently running.
        - Sets `self.update_data_flag` to `False`, indicating that data updates are not currently active.
        - Initializes `self.tracker` with the provided `tracker` object for tracking motion data.
        - Initializes `self.Tracked_Motion_Data` as an empty list to store tracked motion data during the procedure.

        Parameters:
        - **root (Tk)**: The root window of the Tkinter application.
        - **model (object)**: The model used for digit recognition in real-time.
        - **tracker (object)**: The tracker object used for capturing motion data.

        Returns:
        - **None**: This method initializes various attributes and components but does not return any values.

        Example:
        - Creating an instance of this class will set up the main window with the specified title and background color, initialize the digit recognizer, set up the user interface, and prepare components for handling video and motion tracking.
        """
        self.root = root
        self.root.title("Gastro-Check")
        self.style = ttk.Style()
        self.style.theme_use("clam")  # A more modern theme
        self.style.configure("TButton", padding=6, relief="flat", background="#4CAF50", foreground="white", font=("Arial", 12))
        self.style.configure("TLabel", background="white", foreground="black", font=("Arial", 12))
        self.style.configure("TFrame", background="white")

        self.image_size = image_size
        self.labelshift = labelshift
        self.num_classes = num_classes

        self.digit_recognizer = RealTimeDigitRecognition(model, self.image_size, labelshift=labelshift)
        self.setup_ui()

        self.video_feed = VideoFeed(self.root, self.video_frame, self.digit_recognizer, self.array_data_label, self.digit_show_label, self.gi_label, self.num_classes)
        self.start_time = None
        self.running_timer = False
        self.is_procedure_running = False
        self.update_data_flag = False
        self.tracker = tracker
        self.Tracked_Motion_Data = []

    def setup_ui(self):
        """
        Sets up the user interface by creating and arranging various frames and components.

        This method calls the following methods to initialize and place different UI elements:
        
        1. **Creates the Video Frame**:
        - Calls `self.create_video_frame()` to create and configure the frame for displaying video content.

        2. **Creates the Control Frame**:
        - Calls `self.create_control_frame()` to create and configure the frame containing controls for starting and stopping procedures, and displaying the timer.

        3. **Creates the Array Frame**:
        - Calls `self.create_array_frame()` to create and configure the frame for displaying tracked data metrics and other related information.

        4. **Creates the GI Frame**:
        - Calls `self.create_gi_frame()` to create and configure the frame displaying a graphical image related to the application.

        This method is typically called to set up the entire user interface layout when the application starts or when the UI needs to be refreshed.

        Returns:
        - **None**: This method initializes and arranges various UI components but does not return any values.

        Example:
        - Calling this method will result in the arrangement of video display, control buttons, data metrics, and graphical images within the main application window.
        """
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        self.create_video_frame()
        self.create_control_frame()
        self.create_array_frame()
        self.create_gi_frame()

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def create_video_frame(self):
        """
        Creates and configures a frame for displaying video content in the user interface.
        
        This method ensures the video frame has a fixed size so that it does not take up excessive space.
        """
        # Create a frame to hold the video label
        self.video_frame_container = Frame(self.root, width=400, height=300)
        self.video_frame_container.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Use the ttk.Label for the video frame (no width and height options here)
        self.video_frame = ttk.Label(self.video_frame_container)
        self.video_frame.pack(fill="both", expand=True)

    def create_control_frame(self):
        """
        Creates and configures a control frame within the user interface to manage the procedure and display controls.

        This method performs the following actions:

        1. **Creates a Frame**:
        - Initializes a new `Frame` widget named `self.control_frame` with a light sky blue background color (`bg="LightSkyBlue1"`).
        - Places the frame in the grid layout at row 1, column 0, with padding of 10 pixels on all sides (`padx=10, pady=10`).

        2. **Adds a Start Button**:
        - Creates a `Button` widget named `self.start_button` with the text "Start Procedure".
        - Sets the button's command to `self.start_procedure`, which will be called when the button is pressed.
        - Configures the background color to light sky blue (`bg="LightSkyBlue1"`).
        - Packs the button into `self.control_frame`.

        3. **Adds a Timer Label**:
        - Creates a `Label` widget named `self.timer_label` to display the time in the format "Time: 0:00".
        - Sets the font of the label to Arial, size 16.
        - Configures the background color to light sky blue (`bg="LightSkyBlue1"`).
        - Packs the label into `self.control_frame`.

        4. **Adds a Start Again Button**:
        - Creates a `Button` widget named `self.start_again_button` with the text "Start Again".
        - Sets the button's command to `self.start_again`, which will be called when the button is pressed.
        - Configures the background color to light sky blue (`bg="LightSkyBlue1"`).
        - Initially hides the button using `pack_forget()`.

        This method sets up a section of the user interface with controls for starting and restarting the procedure, and displaying the elapsed time.

        Returns:
        - **None**: This method initializes and configures the frame and its child widgets but does not return any values.

        Example:
        - After calling this method, `self.control_frame` will contain a "Start Procedure" button, a timer label displaying "Time: 0:00", and a hidden "Start Again" button.
        """
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.grid(row=1, column=0, padx=10, pady=10)

        self.start_button = ttk.Button(self.control_frame, text="Start Procedure", command=self.start_procedure)
        self.start_button.pack()

        self.timer_label = ttk.Label(self.control_frame, text="Time: 0:00", font=("Arial", 16))
        self.timer_label.pack()

        self.start_again_button = ttk.Button(self.control_frame, text="Start Again", command=self.start_again)
        self.start_again_button.pack_forget()

    def create_array_frame(self):
        """
        Creates and configures a frame within the user interface to display data related to tracked metrics and digits seen.

        This method performs the following actions:

        1. **Creates a Frame**:
        - Initializes a new `Frame` widget named `self.array_frame` with a light sky blue background color (`bg="LightSkyBlue1"`).
        - Places the frame in the grid layout at row 1, column 1, with padding of 10 pixels on all sides (`padx=10, pady=10`).

        2. **Adds a Title Label**:
        - Creates a `Label` widget named `self.array_label` with the text "Digits Seen".
        - Sets the font of the label to Arial, size 14.
        - Configures the background color to light sky blue (`bg="LightSkyBlue1"`).
        - Packs the label into `self.array_frame`.

        3. **Adds a Data Display Label**:
        - Creates another `Label` widget named `self.array_data_label` with an initial empty text.
        - Sets the font of the label to Arial, size 12.
        - Configures the background color to light sky blue (`bg="LightSkyBlue1"`).
        - Packs the label into `self.array_frame`.

        4. **Creates Tracked Data Labels**:
        - Calls `self.create_tracked_data_labels()` to initialize and configure labels for displaying various tracked data metrics within `self.array_frame`.

        This method sets up a section of the user interface dedicated to displaying digits seen and tracked data metrics.

        Returns:
        - **None**: This method initializes and configures the frame and labels but does not return any values.

        Example:
        - After calling this method, `self.array_frame` will contain a title label for "Digits Seen", an empty data label for displaying additional information, and labels for various tracked metrics.
        """
        self.array_frame = ttk.Frame(self.root)
        self.array_frame.grid(row=1, column=1, padx=10, pady=15, sticky="nsew")

        # Create two frames: one for "Digits Seen" and one for tracked data
        self.digits_frame = ttk.Frame(self.array_frame)
        self.digits_frame.grid(row=0, column=0, padx=10, pady=15, sticky="n")

        self.tracked_data_frame = ttk.Frame(self.array_frame)
        self.tracked_data_frame.grid(row=0, column=1, padx=40, pady=10, sticky="n")

        # Styles
        self.style.configure("Blue.TLabel", foreground="blue", font=("Arial", 14))
        self.style.configure("Green.TLabel", foreground="green", font=("Arial", 14))


        # Adding digits information in the digits_frame
        self.digit_label = ttk.Label(self.digits_frame, text="Area", style="Blue.TLabel")
        self.digit_label.pack()

        self.digit_show_label = ttk.Label(self.digits_frame, text="", style="Blue.TLabel")
        self.digit_show_label.pack()

        self.array_label = ttk.Label(self.digits_frame, text="Gastric Area's Seen", style="Green.TLabel")
        self.array_label.pack()

        self.array_data_label = ttk.Label(self.digits_frame, text="", style="Green.TLabel")
        self.array_data_label.pack()

        # Creating the tracked data labels in the tracked_data_frame
        self.create_tracked_data_labels()

    def create_tracked_data_labels(self):
        """
        Creates and initializes a set of labels to display various tracked data metrics in the user interface.

        This method performs the following actions:

        1. **Creates and Configures Labels**:
        - Calls `self.create_data_label()` for each metric type to create labels with initial default values:
            - **Time**: Displays the elapsed time (e.g., "Time: 0 s").
            - **Path Length**: Displays the total path length (e.g., "Path Length: 0 m").
            - **Angular Length**: Displays the total angular displacement (e.g., "Angular Length: 0 °").
            - **Response Orientation**: Displays the final orientation (e.g., "Response Orientation: 0 °").
            - **Depth Perception**: Displays the depth perception (e.g., "Depth Perception: 0 m").
            - **Motion Smoothness**: Displays the smoothness of motion (e.g., "Motion Smoothness: 0 m/s³").
            - **Average Velocity**: Displays the average velocity (e.g., "Average Velocity: 0 mm/s").

        2. **Stores Labels**:
        - Stores the created labels in a dictionary named `self.labels`, where each key corresponds to a metric name (e.g., 'time', 'path_length') and each value is the associated `Label` widget.

        This method sets up the user interface with labels for displaying various metrics related to tracked data, initializing them with default values.

        Returns:
        - **None**: This method initializes and stores the labels in the `self.labels` dictionary but does not return any values.

        Example:
        - After calling this method, `self.labels` will contain `Label` widgets for each tracked data metric, each initialized with default text.
        """
        self.labels = {
            'time': self.create_data_label("Time: 0 s", self.tracked_data_frame),
            'path_length': self.create_data_label("Path Length: 0 m", self.tracked_data_frame),
            'angular_length': self.create_data_label("Angular Length: 0 °", self.tracked_data_frame),
            'response_orientation': self.create_data_label("Response Orientation: 0 °", self.tracked_data_frame),
            'depth_perception': self.create_data_label("Depth Perception: 0 m", self.tracked_data_frame),
            'motion_smoothness': self.create_data_label("Motion Smoothness: 0 m/s³", self.tracked_data_frame),
            'average_velocity': self.create_data_label("Average Velocity: 0 mm/s", self.tracked_data_frame)
        }

    def create_data_label(self, text, parent_frame):
        """
        Creates and configures a label to display text in the user interface.

        This method performs the following actions:

        1. **Initializes a Label**:
        - Creates a `Label` widget with the specified `text` parameter to display.
        - Sets the font of the label to Arial, size 12.
        - Configures the background color of the label to light sky blue (`bg="LightSkyBlue1"`).

        2. **Adds the Label to the Frame**:
        - Packs the label into `self.array_frame` using `pack()`, which arranges it within the frame with default settings.

        3. **Returns the Label**:
        - Returns the created `Label` widget so that it can be further manipulated if needed.

        This method helps in creating standardized labels with a specific appearance for displaying various pieces of information in the user interface.

        Parameters:
        - **text (str)**: The text to be displayed on the label.

        Returns:
        - **Label**: The created `Label` widget configured with the specified text, font, and background color.

        Example:
        - Calling `create_data_label("Sample Data")` creates a label with the text "Sample Data", sets its font and background color, and returns the label widget.
        """
        self.style.configure("Purple.TLabel", foreground="purple", font=("Arial", 13))
        label = ttk.Label(parent_frame, text=text, style="Purple.TLabel")
        label.pack(anchor="w")  # Align the labels to the left inside the parent frame
        return label


    def create_gi_frame(self):
        """
        Creates and configures a frame to display a graphical image in the user interface.
        """
        self.gi_frame = ttk.Frame(self.root)
        self.gi_frame.grid(row=0, column=2, padx=5, pady=5)

        self.gi_image = Image.open("GI-Tract-Images/Seen.png").resize((400, 600), Image.Resampling.LANCZOS)
        self.gi_image_tk = ImageTk.PhotoImage(self.gi_image)

        self.gi_label = ttk.Label(self.gi_frame, image=self.gi_image_tk)
        self.gi_label.pack()

    def update_tracked_data(self):
        """
        Updates the tracked motion data metrics and refreshes the corresponding labels in the user interface.
        """
        timestamp_array, trans_matrix = zip(*self.Tracked_Motion_Data) if self.Tracked_Motion_Data else ([], [])
        metrics = {
            'time': round(self.calculate_time(timestamp_array), 3),
            'path_length': round(self.calculate_path_length(trans_matrix), 3),
            'angular_length': round(self.calculate_angular_length(trans_matrix), 3),
            'response_orientation': round(self.calculate_response_orientation(trans_matrix), 3),
            'depth_perception': round(self.calculate_depth_perception(trans_matrix), 3),
            'motion_smoothness': round(self.calculate_motion_smoothness(trans_matrix, timestamp_array), 3),
            'average_velocity': round(self.calculate_average_velocity(trans_matrix, timestamp_array), 3)
        }
        self.update_labels(metrics)

    def reset_tracked_data_labels(self):
        """
        Resets the labels displaying tracked data metrics in the user interface to their default values.
        """
        metrics = {
            'time': 0,
            'path_length': 0,
            'angular_length': 0,
            'response_orientation': 0,
            'depth_perception': 0,
            'motion_smoothness': 0,
            'average_velocity': 0
        }
        self.update_labels(metrics)

    def update_labels(self, metrics):
        """
        Updates the user interface labels with the latest metric values and their corresponding units.

        This method performs the following actions:

        1. **Iterates through the metrics**: For each key-value pair in the `metrics` dictionary:
        - `key` represents the type of measurement (e.g., 'time', 'path_length').
        - `value` represents the corresponding value of the measurement.
        2. **Finds the appropriate unit**: Calls `self.findUnit(key)` to determine the correct unit for each measurement type.
        3. **Updates label text**:
        - Retrieves the label associated with each metric using `self.labels[key]`.
        - Formats the label text by converting the key to a human-readable form (`key.replace('_', ' ').title()`) and appending the value and its unit.
        - Updates the label with the formatted string.

        This method ensures that the labels in the user interface display the latest metric values along with their corresponding units.

        Parameters:
        - **metrics (dict)**: A dictionary where the keys are metric names (e.g., 'time', 'path_length') and the values are the corresponding measurement values.

        Example:
        - If `metrics = {'time': 12.5, 'average_velocity': 35}`, this method updates the corresponding labels to show:
        - `Time: 12.5 s`
        - `Average Velocity: 35 mm/s`
        """
        for key, value in metrics.items():
            self.labels[key].config(text=f"{key.replace('_', ' ').title()}: {value} {self.findUnit(key)}")

    def findUnit(self, key):
        """
        Returns the appropriate unit of measurement for a given data key.

        This method maps specific keys (representing different measurements) to their corresponding units. It performs the following actions:
        """
        units = {
            'time': 's',
            'path_length': 'm',
            'angular_length': '°',
            'response_orientation': '°',
            'depth_perception': 'm',
            'motion_smoothness': 'm/s³',
            'average_velocity': 'mm/s'
        }
        return units.get(key, 'NaN')

    def update_data_periodically(self):
        """
        Periodically updates the tracked data while the procedure is running.

        This method performs the following actions:

        1. **Checks if data updates are enabled**: 
        - If `self.update_data_flag` is `True`, it proceeds with updating the tracked data.
        2. **Calls the update function**: 
        - Invokes `self.update_tracked_data()` to update the tracked data.
        3. **Schedules the next update**: 
        - Uses `self.root.after(1000, self.update_data_periodically)` to schedule the next update in 1 second (1000 milliseconds).
        - This process continues as long as `self.update_data_flag` remains `True`.

        This method allows for continuous, periodic updates to the tracked data during the procedure.

        Note:
        - This method should be called once to initiate periodic updates. It will continue updating every second until `self.update_data_flag` is set to `False`.
        - Ensure `self.update_tracked_data()` is properly implemented to handle the tracking data updates.
        """
        if self.update_data_flag:
            self.update_tracked_data()
            self.root.after(1000, self.update_data_periodically)

    def start_procedure(self):
        """
        Initiates the digit recognition and motion tracking procedure, updating the user interface and starting background processes.

        This method performs the following actions:

        1. **Starts digit recognition**: Enables updates to recognized digits by setting `self.digit_recognizer.update_digits` to `True`.
        2. **Enables visual indication**: Activates a blinking dot on the video feed by setting `self.video_feed.blinking_dot` to `True`, signaling that the procedure is running.
        3. **Initializes timer**:
        - Records the current time in `self.start_time`.
        - Sets `self.running_timer` to `True` and calls `self.update_timer()` to start updating the timer label every second.
        4. **Starts procedure execution**: Sets `self.is_procedure_running` to `True`, indicating that the procedure has begun.
        5. **Starts data tracking in a new thread**: Begins tracking and saving motion data by starting a new thread that runs `self.save_tracked_data_to_variable()`. This prevents blocking the main application thread.
        6. **Updates data flag**: Sets `self.update_data_flag` to `True`, signaling that data should be periodically updated.
        7. **Configures the start button**: Changes the `start_button` text to "Stop Procedure" and sets its command to `self.stop_procedure`, allowing the user to stop the procedure.
        8. **Periodically updates data**: Calls `self.update_data_periodically()` to ensure regular updates of the tracked data.

        This method sets up all necessary components to begin the digit recognition and motion tracking process, while updating the user interface accordingly.

        """
        self.digit_recognizer.update_digits = True
        self.video_feed.blinking_dot = True

        self.start_time = time.time()
        self.running_timer = True
        self.update_timer()

        self.is_procedure_running = True
        threading.Thread(target=self.save_tracked_data_to_variable).start()
        self.update_data_flag = True
        self.start_button.config(text="Stop Procedure", command=self.stop_procedure)

        self.update_data_periodically()

    def update_timer(self):
        """
        Updates the timer display in the user interface, showing the elapsed time since the procedure started.

        This method performs the following actions:
        
        1. **Checks if the timer is running**: If `self.running_timer` is `True`, the method proceeds to update the timer display.
        2. **Calculates elapsed time**: Determines the time elapsed since the start of the procedure by subtracting `self.start_time` from the current time.
        3. **Formats the time**: Converts the elapsed time into minutes and seconds using the `divmod` function.
        4. **Updates the timer label**: Configures the `self.timer_label` text to show the formatted time as "Time: MM:SS".
        5. **Schedules the next update**: Uses `self.root.after(1000, self.update_timer)` to schedule the next timer update after 1 second (1000 milliseconds).

        This method allows for continuous updating of the timer every second while the procedure is active.

        Note:
        - The method should only be called when `self.running_timer` is `True`.
        - Ensure that `self.start_time` is initialized before calling this method.
        """
        if self.running_timer:
            elapsed_time = time.time() - self.start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            self.timer_label.config(text=f"Time: {minutes}:{seconds:02d}")
            self.root.after(1000, self.update_timer)

    def stop_procedure(self):
        """
        Terminates the digit recognition procedure, updates tracking data, and performs cleanup operations.

        This method handles the following actions when the procedure is stopped:

        1. **Stops digit recognition**: Disables further updates to recognized digits by setting `self.digit_recognizer.update_digits` to `False`.
        2. **Stops video feed indicator**: Turns off any visual indicators of the procedure (such as a blinking dot) by setting `self.video_feed.blinking_dot` to `False`.
        3. **Stops the timer**: Sets `self.running_timer` to `False`, stopping the internal procedure timer.
        4. **Stops procedure execution**: Sets `self.is_procedure_running` to `False`, signaling that the procedure has ended.
        5. **Updates tracked data**: Calls `self.update_tracked_data()` to perform any final updates to the tracked motion data before the procedure stops.
        6. **Disables further data updates**: Sets `self.update_data_flag` to `False` to prevent additional data updates after the procedure has ended.
        7. **Adjusts timer label color**: 
        - Calculates the elapsed time since the procedure started.
        - Changes the color of the timer label to red if the elapsed time is less than 7 minutes; otherwise, sets it to green.
        8. **Stops tracking**: Calls `tracker.stop_tracking()` to terminate the tracking process and finalize the collection of tracking data.
        9. **Saves tracked data**: Calls `self.save_tracked_data_to_file()` to store the tracked motion data in a CSV file for future reference.
        10. **Disables the start button**: Updates the `start_button` text to "Procedure Stopped" and disables it to prevent restarting.
        11. **Displays the "Start Again" button**: Makes the `start_again_button` visible to allow the user to restart the procedure if desired.

        This method ensures a proper shutdown of the digit recognition and tracking procedure while safely saving collected data.
        """
        self.digit_recognizer.update_digits = False
        self.video_feed.blinking_dot = False
        self.running_timer = False
        self.is_procedure_running = False
        self.update_tracked_data()
        self.update_data_flag = False

        elapsed_time = time.time() - self.start_time
        minutes, _ = divmod(int(elapsed_time), 60)

        # Create or modify a style for the timer label
        style = ttk.Style()
        style.configure("Timer.TLabel", foreground="red" if minutes < 7 else "green")
        self.timer_label.config(style="Timer.TLabel")

        tracker.stop_tracking()

        # Prompt the user to save the log file
        should_save = messagebox.askyesno("Save Tracked Data File", "Would you like to save the log file containing the tracked data?")
        if should_save:
            self.save_tracked_data_to_file()

        self.start_button.config(text="Procedure Stopped", state="disabled")
        self.start_again_button.pack()


    def save_tracked_data_to_file(self):
        """
        Saves the tracked motion data to a CSV file.

        This method performs the following actions:
        
        1. **Creates a directory**: Ensures the 'ProcedureData' directory exists by creating it if it does not.
        2. **Generates a timestamp**: Creates a timestamp string based on the current date and time, formatted as 'YYYYMMDD_HHMMSS'.
        3. **Creates a CSV filename**: Constructs the filename for the CSV file using the generated timestamp.
        4. **Writes data to the CSV file**:
            - Opens the CSV file in write mode.
            - Creates a CSV writer object.
            - Iterates over the `self.Tracked_Motion_Data` list, writing each timestamp and corresponding data to the CSV file.

        The method ensures that the motion tracking data collected during the procedure is saved in a structured format, allowing for easy analysis and review.

        Note:
        - Ensure `self.Tracked_Motion_Data` contains data in the expected format before calling this method.
        - The CSV file will be saved in the 'ProcedureData' directory, which will be created if it does not already exist.
        """
        folder_path = 'Data/ProcedureData'
        os.makedirs(folder_path, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = os.path.join(folder_path, f'tracked_motion_data_{timestamp}.csv')

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for (timeStamp, data) in self.Tracked_Motion_Data:
                writer.writerow([timeStamp, data])

    def save_tracked_data_to_variable(self):
        """
        Continuously collects and saves tracking data while the procedure is running.

        This method initiates the tracking process and collects data at regular intervals, storing the data in the `Tracked_Motion_Data` list. 
        It performs the following actions:

        1. Starts tracking: Calls `self.tracker.start_tracking()` to begin the tracking process.
        2. Collects tracking data: Continuously retrieves frames from the tracker while `self.is_procedure_running` is True.
            - Extracts the relevant data from the frame using `self.tracker.get_frame()[3]`.
            - Checks if the data contains any NaN values using `np.isnan(data).any()`.
            - If valid, calculates the inverse of the reference transformation matrix and derives the sensor-to-reference transformation matrix.
            - Calculates the elapsed time since the start of the procedure.
            - Appends a list containing the elapsed time and the sensor-to-reference transformation matrix to `self.Tracked_Motion_Data`.
        3. Pauses briefly: Sleeps for 25 milliseconds (`0.025` seconds) between data collection iterations to control the sampling rate.

        This method is used to collect and store motion tracking data in real-time while the procedure is ongoing.

        Note:
        - Ensure `self.tracker`, `self.is_procedure_running`, `self.start_time`, and `self.Tracked_Motion_Data` are properly initialized before calling this method.
        - The method runs in a continuous loop and should be executed in a separate thread or process to avoid blocking the main thread.

        """
        self.tracker.start_tracking()
        while self.is_procedure_running:
            data = self.tracker.get_frame()[3]
            if not np.isnan(data).any():
                T_ref_inv = np.linalg.inv(data[0])
                T_sensor_rel_ref = np.dot(T_ref_inv, data[1])
                elapsed_time = time.time() - self.start_time
                self.Tracked_Motion_Data.append([elapsed_time, T_sensor_rel_ref])
            time.sleep(0.025)



    def start_again(self):
        """
        Resets the application state to allow for a fresh start of the digit recognition procedure.

        This method clears the previously tracked data and reinitializes the user interface components to their default state. 
        It performs the following actions:
        
        1. Resets the `seen_digits`: Clears the set of digits recognized by the digit recognizer to prepare for a new session.
        2. Clears tracked motion data: Empties the `Tracked_Motion_Data` list to remove any previously tracked movement.
        3. Resets display labels: Updates the `array_data_label` and `timer_label` to their initial state, clearing any information displayed to the user.
        4. Resets tracking data: Invokes the `reset_tracked_data()` method to reset any internal tracking mechanisms.
        5. Configures the start button: Sets the `start_button` text back to "Start Procedure" and re-enables it, allowing the user to begin the process again.
        6. Hides the "Start Again" button: Calls `pack_forget()` on the `start_again_button` to remove it from the user interface.
        """
        self.digit_recognizer.seen_digits = set()
        self.Tracked_Motion_Data = []
        self.array_data_label.config(text="")

        # Use ttk.Style to configure the timer label color instead of fg
        style = ttk.Style()
        style.configure("Timer.TLabel", foreground="black")
        self.timer_label.config(text="Time: 0:00", style="Timer.TLabel")

        self.reset_tracked_data_labels()
        self.start_button.config(text="Start Procedure", state="normal", command=self.start_procedure)
        self.start_again_button.pack_forget()


    def on_close(self):
        """
        Handles the cleanup and shutdown process when closing the application window.

        This method performs three main tasks:
        1. Stops the video feed:
        2. Destroys the root window:
        3. Closes the tracker:
        """
        self.video_feed.stop()
        self.root.destroy()
        self.tracker.close()


    # Motion Metric Calculators
    def calculate_time(self, timeStamps):
        if len(timeStamps) > 1:
            result = timeStamps[-1] - timeStamps[0]
        else:
            result = 0  # or handle the case appropriately, maybe setting result to 0 or some default value

        result = round(result*10)/10
        return result

    def calculate_path_length(self, Tvector):
        result = 0
        for i in range(1,len(Tvector)):
            result = result + math.sqrt((Tvector[i][0][3]-Tvector[i - 1][0][3])**2+(Tvector[i][1][3]-Tvector[i - 1][1][3])**2+(Tvector[i][2][3]-Tvector[i - 1][2][3])**2)
        result = result/1000
        result = round(result*100)/100
        return result

    def calculate_angular_length(self, Tvector):
        resultXY=0
        for i in range(1,len(Tvector)):
            temp=self.calculateAngleDistance(Tvector[i],Tvector[i - 1])
            resultXY=resultXY + math.sqrt(temp[0]**2+temp[1]**2)
        return resultXY

    def calculateAngleDistance(self, T1,T2):
        angles = [0,0,0]
        R11=T2[0][0] * T1[0][0] + T2[0][1] * T1[0][1] + T2[0][2] * T1[0][2]
        R21=T2[1][0] * T1[0][0] + T2[1][1] * T1[0][1] + T2[1][2] * T1[0][2]
        R31=T2[2][0] * T1[0][0] + T2[2][1] * T1[0][1] + T2[2][2] * T1[0][2]
        R32=T2[2][0] * T1[1][0] + T2[2][1] * T1[1][1] + T2[2][2] * T1[1][2]
        R33=T2[2][0] * T1[2][0] + T2[2][1] * T1[2][1] + T2[2][2] * T1[2][2]
        angles[0]=math.atan2(R21,R11)
        angles[1]=math.atan2(- R31,math.sqrt(R32 * R32 + R33 * R33))
        angles[2]=math.atan2(R32,R33)
        return angles

    def calculate_response_orientation(self, Tvector):
        result=0
        for i in range(1,len(Tvector)):
            temp=self.calculateAngleDistance(Tvector[i],Tvector[i - 1])
            result=result + math.fabs(temp[2])
        return result

    def calculate_depth_perception(self, Tvector):
        result = 0
        for i in range(1,len(Tvector)):
            result = result + math.fabs((Tvector[i][2][3]-Tvector[i - 1][2][3]))
        return result/1000

    def calculate_motion_smoothness(self, Tvector, timeStamps):
        if len(timeStamps) < 1:
            return 0
        
        T=timeStamps[-1];
        d1x_dt1=[]; d1y_dt1=[]; d1z_dt1=[]; deltaT = []; timeStampsNew = [];
        for i in range(1,len(Tvector)):
            deltaT.append(timeStamps[i]-timeStamps[i-1])
            d1x_dt1.append((Tvector[i][0][3] - Tvector[i-1][0][3]) / deltaT[i-1])
            d1y_dt1.append((Tvector[i][1][3] - Tvector[i-1][1][3]) / deltaT[i-1])
            d1z_dt1.append((Tvector[i][2][3] - Tvector[i-1][2][3]) / deltaT[i-1])
            timeStampsNew.append((timeStamps[i]+timeStamps[i-1])/2)
        timeStamps = timeStampsNew

        d2x_dt2=[]; d2y_dt2=[]; d2z_dt2=[]; deltaT = []; timeStampsNew = [];
        for i in range(1,len(d1x_dt1)):
            deltaT.append(timeStamps[i]-timeStamps[i-1])
            d2x_dt2.append((d1x_dt1[i] - d1x_dt1[i-1]) / deltaT[i-1])
            d2y_dt2.append((d1y_dt1[i] - d1y_dt1[i-1]) / deltaT[i-1])
            d2z_dt2.append((d1z_dt1[i] - d1z_dt1[i-1]) / deltaT[i-1])
            timeStampsNew.append((timeStamps[i]+timeStamps[i-1])/2)
        timeStamps = timeStampsNew

        d3x_dt3=[]; d3y_dt3=[]; d3z_dt3=[]; deltaT = []; timeStampsNew = [];
        for i in range(1,len(d2x_dt2)):
            deltaT.append(timeStamps[i]-timeStamps[i-1])
            d3x_dt3.append((d2x_dt2[i] - d2x_dt2[i-1]) / deltaT[i-1])
            d3y_dt3.append((d2y_dt2[i] - d2y_dt2[i-1]) / deltaT[i-1])
            d3z_dt3.append((d2z_dt2[i] - d2z_dt2[i-1]) / deltaT[i-1])
            timeStampsNew.append((timeStamps[i]+timeStamps[i-1])/2)
        timeStamps = timeStampsNew

        j = [(x**2 + y**2 +z**2) for x, y ,z in zip(d3x_dt3, d3y_dt3, d3z_dt3)]
        MS = math.sqrt((1 / (2*T)) * np.trapz(j,timeStamps))
        return MS*10**6     # m/s^3

    def calculateVelocity(self, Tvector,timeStamps):
        velocity = []
        for i in range(1,len(Tvector)):
            distance = math.sqrt((Tvector[i][0][3]-Tvector[i - 1][0][3])**2+(Tvector[i][1][3]-Tvector[i - 1][1][3])**2+(Tvector[i][2][3]-Tvector[i - 1][2][3])**2)
            deltaT = timeStamps[i]-timeStamps[i-1]
            velocity.append(distance / deltaT * 1000)
        return velocity # mm/s

    def calculate_average_velocity(self, Tvector,timeStamps):
        if len(timeStamps) < 1:
            return 0
        meanVelocity = np.mean(self.calculateVelocity(Tvector,timeStamps))
        return meanVelocity
   
# Main Application Entry
if __name__ == "__main__":
    # Load your trained CNN model
    model = load_model('Data/models/Model_Training_Images_Colour_ImageSize(100, 100, 3).h5')  # Replace with your model file
    labelshift = False # Change to True if a labelshift of (+1) is used in the training data (if the training data contains a -1 class)
    num_classes= 4

    # Set up the tracker
    settings_aurora = {
        "tracker type": "aurora",
        "serial port": 4, # Specify port 4 explicitly
        "verbose": True,
    }
    tracker = NDITracker(settings_aurora)

    # Call the App
    root = tk.Tk()
    app = Application(root, model, tracker, image_size=(100, 100, 3), labelshift=labelshift, num_classes=num_classes) # Replace with your image size of file
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()