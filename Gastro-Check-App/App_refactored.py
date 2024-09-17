import os
import math
import csv
import time
import threading
from datetime import datetime
from functools import partial

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Frame, Button

from tensorflow.keras.models import load_model # type: ignore
from sksurgerynditracker.nditracker import NDITracker
import six



class RealTimeDigitRecognition:
    def __init__(self, model, frame_skip=3, downscale_factor=0.5):
        self.model = model
        self.frame_skip = frame_skip
        self.downscale_factor = downscale_factor
        self.seen_digits = set()  # Use set for faster lookups
        self.frame_count = 0
        self.update_digits = False

    def preprocess_frame(self, frame):
        """Resizes and converts the frame to grayscale for model prediction."""
        frame = cv2.resize(frame, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (28, 28))
        return resized_frame.reshape(1, 28, 28, 1).astype('float32') / 255

    def predict_digit(self, processed_frame):
        """Runs the model prediction and returns the predicted class."""
        predictions = self.model.predict(processed_frame)
        predicted_class = np.argmax(predictions)
        return predicted_class, predictions

    def process_frame(self, frame):
        """Processes a single frame and updates the seen digits list if required."""
        processed_frame = self.preprocess_frame(frame)
        predicted_class, predictions = self.predict_digit(processed_frame)

        # Update seen digits if the process is running and the digit is valid
        if self.update_digits and predicted_class != 6 and predicted_class not in self.seen_digits:
            self.seen_digits.add(predicted_class)

        return predictions, predicted_class

    def display_frame(self, frame, predicted_class, blinking_dot):
        """Displays the predicted digit and a blinking dot if the process is running."""
        if predicted_class != -1:
            cv2.putText(frame, f"Predicted Digit: {predicted_class}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if blinking_dot:
            cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)  # Red dot at the top-left corner


# With frame skipping, less computational load
class VideoFeedWithFrameSkipping:
    def __init__(self, window, video_frame, digit_recognizer, array_data_label):
        self.window = window
        self.video_frame = video_frame
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.digit_recognizer = digit_recognizer
        self.array_data_label = array_data_label
        self.frame_skip = 3
        self.frame_count = 0
        self.blinking_dot = False  # Blinking red dot control

        self.update_video()

    def update_video(self):
        """Update video frame and run digit recognition."""
        if self.running:
            ret, frame = self.cap.read()
            predicted_class = -1  # Initialize predicted_class with a default value (e.g., -1 for no prediction)
            
            if ret:
                self.frame_count += 1
                if self.frame_count % self.frame_skip == 0:
                    # Process frame and get the predicted class
                    _, predicted_class = self.digit_recognizer.process_frame(frame)

                # Display the frame with the predicted class and blinking dot
                self.digit_recognizer.display_frame(frame, predicted_class, self.blinking_dot)

                # Convert frame to display in Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.config(image=imgtk)

                # Only update the "Digits Seen" label when the procedure is running
                if self.digit_recognizer.update_digits:
                    self.update_seen_digits_display()

            # Continue updating after a short delay
            self.window.after(10, self.update_video)

# Without frame skipping, high computational load
class VideoFeed:
    def __init__(self, window, video_frame, digit_recognizer, array_data_label):
        self.window = window
        self.video_frame = video_frame
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.digit_recognizer = digit_recognizer
        self.array_data_label = array_data_label
        self.blinking_dot = False  # Control blinking red dot for display
        self.frame_update_delay = 10000  # Delay in ms for video frame updates

        self.start_video_feed()

    def start_video_feed(self):
        """Starts the video feed and handles real-time digit recognition."""
        if self.running:
            ret, frame = self.read_frame()
            if ret:
                predicted_class = self.process_frame_and_recognize_digit(frame)
                # self.display_frame(frame, predicted_class)

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
        _, predicted_class = self.digit_recognizer.process_frame(frame)
        return predicted_class

    # def display_frame(self, frame, predicted_class):
    #     """Displays the processed frame in the UI with the predicted class and optional blinking dot."""
    #     self.digit_recognizer.display_frame(frame, predicted_class, self.blinking_dot)
    #     frame_rgb = self.convert_frame_to_rgb(frame)
    #     self.update_tkinter_image(frame_rgb)

    def convert_frame_to_rgb(self, frame):
        """Converts the frame to RGB format."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # def update_tkinter_image(self, frame_rgb):
    #     """Converts a frame to Tkinter-compatible format and updates the video display."""
    #     img = Image.fromarray(frame_rgb)
    #     imgtk = ImageTk.PhotoImage(image=img)
    #     self.video_frame.imgtk = imgtk
    #     self.video_frame.config(image=imgtk)

    def update_seen_digits_display(self):
        """Updates the label that displays the seen digits."""
        seen_digits_text = ", ".join(map(str, self.digit_recognizer.seen_digits))
        self.array_data_label.config(text=seen_digits_text)

    def stop(self):
        """Stops the video feed and releases the camera."""
        self.running = False
        self.cap.release()


class Application:
    def __init__(self, root, model, tracker):
        self.root = root
        self.root.title("Gastro-Check")
        self.root.configure(bg="LightSkyBlue1")

        self.digit_recognizer = RealTimeDigitRecognition(model)
        self.setup_ui()

        self.video_feed = VideoFeed(self.root, self.video_frame, self.digit_recognizer, self.array_data_label)
        self.start_time = None
        self.running_timer = False
        self.is_procedure_running = False
        self.update_data_flag = False
        self.tracker = tracker
        self.Tracked_Motion_Data = []

    def setup_ui(self):
        self.create_video_frame()
        self.create_control_frame()
        self.create_array_frame()
        self.create_gi_frame()

    def create_video_frame(self):
        self.video_frame = Label(self.root)
        self.video_frame.grid(row=0, column=0, columnspan=2)

    def create_control_frame(self):
        self.control_frame = Frame(self.root, bg="LightSkyBlue1")
        self.control_frame.grid(row=1, column=0, padx=10, pady=10)

        self.start_button = Button(self.control_frame, text="Start Procedure", command=self.start_procedure, bg="LightSkyBlue1")
        self.start_button.pack()

        self.timer_label = Label(self.control_frame, text="Time: 0:00", font=("Arial", 16), bg="LightSkyBlue1")
        self.timer_label.pack()

        self.start_again_button = Button(self.control_frame, text="Start Again", command=self.start_again, bg="LightSkyBlue1")
        self.start_again_button.pack_forget()

    def create_array_frame(self):
        self.array_frame = Frame(self.root, bg="LightSkyBlue1")
        self.array_frame.grid(row=1, column=1, padx=10, pady=10)

        self.array_label = Label(self.array_frame, text="Digits Seen", font=("Arial", 14), bg="LightSkyBlue1")
        self.array_label.pack()

        self.array_data_label = Label(self.array_frame, text="", font=("Arial", 12), bg="LightSkyBlue1")
        self.array_data_label.pack()

        self.create_tracked_data_labels()

    def create_tracked_data_labels(self):
        self.labels = {
            'time': self.create_data_label("Time: VariableWithNamePathLength"),
            'path_length': self.create_data_label("Path Length: VariableWithNamePathLength"),
            'angular_length': self.create_data_label("Angular Length: VariableWithNameAngularLength"),
            'response_orientation': self.create_data_label("Response Orientation: VariableWithNameResponseOrientation"),
            'depth_perception': self.create_data_label("Depth Perception: VariableWithNameDepthPerception"),
            'motion_smoothness': self.create_data_label("Motion Smoothness: VariableWithNameMotionSmoothness"),
            'average_velocity': self.create_data_label("Average Velocity: VariableWithNameAverageVelocity")
        }

    def create_data_label(self, text):
        label = Label(self.array_frame, text=text, font=("Arial", 12), bg="LightSkyBlue1")
        label.pack()
        return label

    def create_gi_frame(self):
        self.gi_frame = Frame(self.root, bg="LightSkyBlue1")
        self.gi_frame.grid(row=0, column=2, padx=5, pady=5)

        self.gi_image = Image.open("gi_tract_model.png").resize((400, 600), Image.Resampling.LANCZOS)
        self.gi_image_tk = ImageTk.PhotoImage(self.gi_image)

        self.gi_label = Label(self.gi_frame, image=self.gi_image_tk)
        self.gi_label.pack()

    def update_tracked_data(self):
        timestamp_array, trans_matrix = zip(*self.Tracked_Motion_Data) if self.Tracked_Motion_Data else ([], [])
        metrics = {
            'time': self.calculate_time(timestamp_array),
            'path_length': self.calculate_path_length(trans_matrix),
            'angular_length': self.calculate_angular_length(trans_matrix),
            'response_orientation': self.calculate_response_orientation(trans_matrix),
            'depth_perception': self.calculate_depth_perception(trans_matrix),
            'motion_smoothness': self.calculate_motion_smoothness(trans_matrix, timestamp_array),
            'average_velocity': self.calculate_average_velocity(trans_matrix, timestamp_array)
        }
        self.update_labels(metrics)

    def update_labels(self, metrics):
        for key, value in metrics.items():
            self.labels[key].config(text=f"{key.replace('_', ' ').title()}: {value}")

    def update_data_periodically(self):
        if self.update_data_flag:
            self.update_tracked_data()
            self.root.after(1000, self.update_data_periodically)

    def start_procedure(self):
        self.digit_recognizer.update_digits = True
        self.video_feed.blinking_dot = True

        self.start_time = time.time()
        self.running_timer = True
        self.update_timer()

        self.is_procedure_running = True
        threading.Thread(target=self.print_tracker_data).start()
        self.update_data_flag = True
        self.start_button.config(text="Stop Procedure", command=self.stop_procedure)

        self.update_data_periodically()

    def update_timer(self):
        if self.running_timer:
            elapsed_time = time.time() - self.start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            self.timer_label.config(text=f"Time: {minutes}:{seconds:02d}")
            self.root.after(1000, self.update_timer)

    def stop_procedure(self):
        self.digit_recognizer.update_digits = False
        self.video_feed.blinking_dot = False
        self.running_timer = False
        self.is_procedure_running = False
        self.update_tracked_data()
        self.update_data_flag = False

        elapsed_time = time.time() - self.start_time
        minutes, _ = divmod(int(elapsed_time), 60)
        self.timer_label.config(fg="red" if minutes < 7 else "green")

        tracker.stop_tracking()
        self.save_tracked_data()
        self.start_button.config(text="Procedure Stopped", state="disabled")
        self.start_again_button.pack()

    def save_tracked_data(self):
        folder_path = 'ProcedureData'
        os.makedirs(folder_path, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = os.path.join(folder_path, f'tracked_motion_data_{timestamp}.csv')

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for (timeStamp, data) in self.Tracked_Motion_Data:
                writer.writerow([timeStamp, data])

    def print_tracker_data(self):
        self.tracker.start_tracking()
        while self.is_procedure_running:
            data = self.tracker.get_frame()[3]
            if not np.isnan(data).any():
                T_ref_inv = np.linalg.inv(data[0])
                T_sensor_rel_ref = np.dot(T_ref_inv, data[1])
                elapsed_time = time.time() - self.start_time
                self.Tracked_Motion_Data.append([elapsed_time, T_sensor_rel_ref])
            #time.sleep(0.5)

    def start_again(self):
        self.digit_recognizer.seen_digits = set()  # Use set for faster lookups
        self.Tracked_Motion_Data = []
        self.array_data_label.config(text="")
        self.timer_label.config(text="Time: 0:00", fg="black")
        self.start_button.config(text="Start Procedure", state="normal", command=self.start_procedure)
        self.start_again_button.pack_forget()

    def on_close(self):
        self.video_feed.stop()
        self.root.destroy()
        self.tracker.close()

    # Motion Metric Calculators
    def calculate_time(self, timestamps):
        return round(timestamps[-1] - timestamps[0], 1) if timestamps else 0

    def calculate_path_length(self, Tvector):
        return round(sum(math.sqrt(sum((Tvector[i][j][3] - Tvector[i-1][j][3])**2 for j in range(3))) for i in range(1, len(Tvector))) / 1000, 2)

    def calculate_angular_length(self, Tvector):
        return sum(math.sqrt(sum(self.calculate_angle_distance(Tvector[i], Tvector[i-1])[:2]**2)) for i in range(1, len(Tvector)))

    def calculate_angle_distance(self, T1, T2):
        R11, R21, R31 = np.dot(T2[:3, :3], T1[:3, :3].T).flat[:3]
        R32, R33 = np.dot(T2[2, :3], T1[1:, :3].T).flat[:2]
        return np.array([math.atan2(R21, R11), math.atan2(-R31, math.hypot(R32, R33)), math.atan2(R32, R33)])

    def calculate_response_orientation(self, Tvector):
        return sum(abs(self.calculate_angle_distance(Tvector[i], Tvector[i-1])[2]) for i in range(1, len(Tvector)))

    def calculate_depth_perception(self, Tvector):
        return round(sum(abs(Tvector[i][2, 3] - Tvector[i-1][2, 3]) for i in range(1, len(Tvector))) / 1000, 2)

    def calculate_motion_smoothness(self, Tvector, timestamps):
        return sum(math.sqrt(sum(self.calculate_angle_distance(Tvector[i], Tvector[i-1])**2)) for i in range(1, len(Tvector)))

    def calculate_average_velocity(self, Tvector, timestamps):
        time_elapsed = timestamps[-1] - timestamps[0] if timestamps else 1
        total_distance = self.calculate_path_length(Tvector)
        return round(total_distance / time_elapsed, 2)

# Main Application Entry
if __name__ == "__main__":
    # Load your trained CNN model
    model_dir = 'models'
    model_name = 'my_digit_classifier_with_no_number_class.h5'
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path)  # Replace with your model file

    settings_aurora = {
        "tracker type": "aurora",
        "ports to probe": 40,
        "verbose": True,
    }

    tracker = NDITracker(settings_aurora)

    root = tk.Tk()
    app = Application(root, model, tracker)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()