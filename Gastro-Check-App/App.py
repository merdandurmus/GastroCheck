import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import time
from tensorflow.keras.models import load_model
import six
from sksurgerynditracker.nditracker import NDITracker
from functools import partial
import threading  # Import threading for continuous tasks
import csv
from datetime import datetime


class RealTimeDigitRecognition:
    def __init__(self, model, frame_skip=3, downscale_factor=0.5):
        self.model = model
        self.frame_skip = frame_skip
        self.downscale_factor = downscale_factor
        self.seen_digits = []  # Initialize seen digits as an empty list
        self.frame_count = 0
        self.update_digits = False  # Flag to control when digits should be updated

    def process_frame(self, frame):
        frame = cv2.resize(frame, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (28, 28))
        resized_frame = resized_frame.reshape(1, 28, 28, 1).astype('float32') / 255

        predictions = self.model.predict(resized_frame)
        predicted_class = np.argmax(predictions)

        # Only update seen digits if the procedure is running
        if self.update_digits and predicted_class != 6 and predicted_class not in self.seen_digits:
            self.seen_digits.append(predicted_class)

        return predictions, predicted_class

    def display_frame(self, frame, predicted_class, blinking_dot):
        # Draw the predicted class at the bottom of the frame only if it's not -1
        if predicted_class != -1:
            cv2.putText(frame, f"Predicted Digit: {predicted_class}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Add a blinking red dot if procedure is running
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
        self.blinking_dot = False  # Blinking red dot control

        self.update_video()

    def update_video(self):
        """Update video frame and run digit recognition."""
        if self.running:
            ret, frame = self.cap.read()
            predicted_class = -1  # Initialize predicted_class with a default value (e.g., -1 for no prediction)
            
            if ret:
                # Process the frame in real-time for digit recognition
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


    def update_seen_digits_display(self):
        """Update the label to display the seen digits."""
        seen_digits_text = ", ".join(map(str, self.digit_recognizer.seen_digits))
        self.array_data_label.config(text=seen_digits_text)

    def stop(self):
        """Stop the video feed."""
        self.running = False
        self.cap.release()


class Application:
    def __init__(self, root, model, tracker):
        self.root = root
        self.root.title("Gastro-Check")

        # Set the background color of the entire window
        self.root.configure(bg="LightSkyBlue1")

        # Create a RealTimeDigitRecognition instance with the trained model
        self.digit_recognizer = RealTimeDigitRecognition(model)

        # Part 1: Video feed frame
        self.video_frame = tk.Label(self.root)
        self.video_frame.grid(row=0, column=0, columnspan=2)

        # Part 2: Procedure Control Buttons and Timer
        self.control_frame = tk.Frame(self.root, bg="LightSkyBlue1") 
        self.control_frame.grid(row=1, column=0, padx=10, pady=10)

        self.start_button = tk.Button(self.control_frame, text="Start Procedure", command=partial(self.start_procedure, tracker), bg="LightSkyBlue1", activebackground="deep pink", borderwidth=0, highlightthickness=0)
        self.start_button.pack()

        self.timer_label = tk.Label(self.control_frame, text="Time: 0:00", font=("Arial", 16), bg="LightSkyBlue1")
        self.timer_label.pack()

        # Start Again button, initially hidden
        self.start_again_button = tk.Button(self.control_frame, text="Start Again", command=partial(self.start_again, tracker), bg="LightSkyBlue1", activebackground="deep pink", borderwidth=0, highlightthickness=0)
        self.start_again_button.pack_forget()  # Hide the button initially

        # Part 3: Display Array Data (seen digits)
        self.array_frame = tk.Frame(self.root, bg="LightSkyBlue1")
        self.array_frame.grid(row=1, column=1, padx=10, pady=10)

        self.array_label = tk.Label(self.array_frame, text="Digits Seen", font=("Arial", 14), bg="LightSkyBlue1")
        self.array_label.pack()

        self.array_data_label = tk.Label(self.array_frame, text="", font=("Arial", 12), bg="LightSkyBlue1")
        self.array_data_label.pack()

        # Part 4: Display GI tract illustration
        self.gi_frame = tk.Frame(self.root, bg="LightSkyBlue1")
        self.gi_frame.grid(row=0, column=2, padx=5, pady=5)

        # Load and display the GI tract image
        self.gi_image = Image.open("gi_tract_model.png")  # Replace with your actual image path
        self.gi_image = self.gi_image.resize((400, 600), Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality downscaling
        self.gi_image_tk = ImageTk.PhotoImage(self.gi_image)


        self.gi_label = tk.Label(self.gi_frame, image=self.gi_image_tk)
        self.gi_label.pack()

        # Initialize VideoFeed with RealTimeDigitRecognition and update seen digits
        self.video_feed = VideoFeed(self.root, self.video_frame, self.digit_recognizer, self.array_data_label)

        # For timer thread
        self.start_time = None
        self.running_timer = False

        self.is_procedure_running = False
        self.print_thread = None  # Thread for continuous printing

        # Initialize the NDI tracker
        self.tracker = tracker

        # Initialize Tracked_Motion_Data array
        self.Tracked_Motion_Data = []

    def start_procedure(self, tracker):
        """Start the procedure and the timer."""
        self.digit_recognizer.update_digits = True  # Enable updating of seen digits
        self.video_feed.blinking_dot = True  # Enable the blinking red dot

        self.start_time = time.time()
        self.running_timer = True
        self.update_timer()

        self.is_procedure_running = True
        self.print_thread = threading.Thread(target=self.print_tracker_data)
        self.print_thread.start()

        self.start_button.config(text="Stop Procedure", command=partial(self.stop_procedure, tracker))

    def update_timer(self):
        """Update the timer every second."""
        if self.running_timer:
            elapsed_time = time.time() - self.start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            self.timer_label.config(text=f"Time: {minutes}:{seconds:02d}")
            self.root.after(1000, self.update_timer)

    def stop_procedure(self, tracker):
        """Stop the procedure and display the time result."""
        self.digit_recognizer.update_digits = False  # Stop updating seen digits
        self.video_feed.blinking_dot = False  # Disable the blinking red dot
        self.running_timer = False

        elapsed_time = time.time() - self.start_time
        minutes, _ = divmod(int(elapsed_time), 60)

        if minutes < 7:
            self.timer_label.config(fg="red")
        else:
            self.timer_label.config(fg="green")
        

        self.is_procedure_running = False
        if self.print_thread:
                self.print_thread.join()
        tracker.stop_tracking()

        # Define the folder path
        folder_path = 'ProcedureData'

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save Tracked_Motion_Data to a CSV file
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = os.path.join(folder_path, f'tracked_motion_data_{timestamp}.csv')
        self.save_data_to_csv(csv_filename)

        self.start_button.config(text="Procedure Stopped", state="disabled")
        self.start_again_button.pack()  # Show the "Start Again" button

    def print_tracker_data(self):
        tracker.start_tracking()
        """Continuously print tracker data while the procedure is running."""
        while self.is_procedure_running:
            data = tracker.get_frame()[3]
            T_ref = data[0]
            T_sensor = data[1]
            T_ref_inv = np.linalg.inv(T_ref)
            T_sensor_rel_ref = np.dot(T_ref_inv, T_sensor)
            elapsedTime = time.time() - self.start_time
            six.print_(T_sensor_rel_ref, elapsedTime)
            self.Tracked_Motion_Data.append([elapsedTime, T_sensor_rel_ref])  # Append data to Tracked_Motion_Data
            time.sleep(0.5)  # Adjust the sleep time as needed   

    def save_data_to_csv(self, filename):
        """
        Save Tracked_Motion_Data to a CSV file, skipping the first entry.

        Args:
            filename (str): The path to the CSV file where data will be saved.

        Notes:
            The CSV file will contain two columns:
            - Timestamp
            - InstrumentPositionRelative2Reference

            The first entry in `self.Tracked_Motion_Data` is skipped.
        """
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Optionally write header row (uncomment if needed)
            # writer.writerow(['Timestamp', 'InstrumentPositionRelative2Reference'])
            
            # Iterate over `Tracked_Motion_Data` starting from the second entry
            for i, (timeStamp, data) in enumerate(self.Tracked_Motion_Data):
                if i == 0:
                    # Skip the first entry
                    continue
                writer.writerow([timeStamp, data])


    def start_again(self, tracker):
        """Reset the application state to start again."""
        self.digit_recognizer.seen_digits = []  # Clear seen digits
        self.Tracked_Motion_Data = [] # Clear tracked data
        self.array_data_label.config(text="")  # Reset displayed seen digits
        self.timer_label.config(text="Time: 0:00", fg="black")  # Reset the timer label
        self.start_button.config(text="Start Procedure", state="normal", command=partial(self.start_procedure, tracker))  # Re-enable the start button
        self.start_again_button.pack_forget()  # Hide the "Start Again" button

    def on_close(self):
        """Handle window close event."""
        self.video_feed.stop()
        self.root.destroy()
        tracker.close()
    

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