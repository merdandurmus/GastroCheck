import os
import cv2
import numpy as np
from PIL import Image, ImageTk


class VideoFeed:
    def __init__(self, window, video_frame, digit_recognizer, inside_outside_recognizer, areas_seen_data_label, current_area_data_label, areas_to_be_seen_data_label, inside_outside_data_label, gi_label, num_classes, video_port):
        """
        Initializes the VideoFeed object for capturing and displaying video frames.

        Parameters:
        - **window (Tk)**: The root Tkinter window where the video feed will be displayed.
        - **video_frame (Label)**: The label widget where video frames will be shown.
        - **digit_recognizer (RealTimeDigitRecognition)**: The digit recognition system to process video frames.
        - **areas_seen_data_label (Label)**: The label widget for displaying recognized digits.

        Initializes the video capture, sets up control flags, and starts the video feed.

        Returns:
        - **None**: This method sets up the video capture and related settings but does not return any values.
        """
        self.window = window
        self.video_frame = video_frame
        self.gi_label = gi_label
        self.video_port = video_port
        self.cap = cv2.VideoCapture(video_port) #1 FOR VIDEO CAPTURE
        self.running = True
        self.digit_recognizer = digit_recognizer
        self.inside_outside_recognizer = inside_outside_recognizer
        self.areas_seen_data_label = areas_seen_data_label
        self.current_area_data_label = current_area_data_label
        self.areas_to_be_seen_data_label = areas_to_be_seen_data_label
        self.inside_outside_data_label = inside_outside_data_label
        self.frame_update_delay = 100  # Delay in ms for video frame updates
        self.num_classes = num_classes

        self.start_video_feed()

    def start_video_feed(self):
        """Starts the video feed and handles real-time digit recognition."""        
        if self.running:
            ret, frame = self.read_frame()
            if ret:
                predicted_loc = self.process_frame_and_recognize_loc(frame)
                predicted_inside_outside = self.process_frame_and_recognize_inside_outside(frame)
                self.display_frame(frame)
                self.update_detecting_digits_display(predicted_loc)
                self.update_detecting_inside_outside_display(predicted_inside_outside)

                # Update seen digits display if procedure is running
                if self.digit_recognizer.update_digits:
                    self.update_seen_digits_display()
                
            # Schedule the next video frame update
            self.window.after(self.frame_update_delay, self.start_video_feed)

    def read_frame(self):
        """Reads a frame from the video feed."""
        return self.cap.read()

    def process_frame_and_recognize_loc(self, frame):
        """Processes the frame for digit recognition and returns the predicted class."""
        predicted_class = self.digit_recognizer.process_frame(frame)
        return predicted_class
    
    def process_frame_and_recognize_inside_outside(self, frame):
        """Processes the frame for digit recognition and returns the predicted class."""
        predicted_class = self.inside_outside_recognizer.process_frame(frame)
        return predicted_class

    def display_frame(self, frame):
        """Displays the processed frame in the UI."""
        frame_rgb = self.convert_frame_to_rgb(frame)
        self.update_tkinter_image(frame_rgb)

    def convert_frame_to_rgb(self, frame):
        """Converts the frame to RGB format."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def update_tkinter_image(self, frame_rgb):
        """Converts a frame to Tkinter-compatible format and updates the video display."""
        img = Image.fromarray(frame_rgb)
        
        width, height = img.size
        
        # Calculate the amount to crop from each side (1/4 of the width)
        crop_amount = width // 9

        # Define the cropping box: (left, upper, right, lower)
        left = crop_amount
        right = width - crop_amount
        top = 0
        bottom = height
        
        img_cropped = img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize((1200, 700), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img_resized)
        self.video_frame.imgtk = imgtk
        self.video_frame.config(image=imgtk)
        
    def digit2gastric(self, digit):
        digit_map = {
            0: "Esophagus (Yellow)",
            1: "Cardia (Blue)",
            2: "Fundus (Red)",
            3: "Body (Greater Curvature) (Orange)",
            4: "Incisura Angularis  (Black)",
            5: "Antrum (Green)",
            6: "Duodenum (Purple)",
        }
        return digit_map.get(digit, "No Anatomical Landmark Present")
    
    def digit2inside_outside(self, digit):
        digit_map = {
            0: "Outside",
            1: "Inside",
        }
        return digit_map.get(digit, "No Recognition Possible")
    
    def combine_images(self, img1, img2):
        """
        Combines two images, preferring green pixels over white pixels, in a more efficient way.
        """
        # Ensure images are in RGBA format
        img1 = img1.convert("RGBA")
        img2 = img2.convert("RGBA")
        
        # Convert images to numpy arrays
        array1 = np.array(img1)
        array2 = np.array(img2)
        
        # Extract RGB channels for both images
        r1, g1, b1, a1 = array1[..., 0], array1[..., 1], array1[..., 2], array1[..., 3]
        r2, g2, b2, a2 = array2[..., 0], array2[..., 1], array2[..., 2], array2[..., 3]
        
        # Create a mask for when to select pixel2
        # Prefer pixel2 if it is green and not white
        mask = (r2 != 255) | (g2 != 255) | (b2 != 255)  # Not white
        mask &= (g2 > r2) & (g2 > b2)  # Green is dominant
        
        # Create the combined image using the mask
        combined_array = np.where(mask[..., None], array2, array1)
        
        # Convert the combined array back to an image
        combined_image = Image.fromarray(combined_array, "RGBA")
        
        return combined_image



    def update_seen_digits_display(self):
        """Updates the label that displays the seen digits."""
        
        # Show all seen areas
        seen_areas_text = "\n".join(map(self.digit2gastric, self.digit_recognizer.seen_digits))
        self.areas_seen_data_label.config(text=seen_areas_text)
        
        # Check which areas have not been seen and show them
        all_digits = set(np.arange(self.num_classes))
        unseen_digits = all_digits - self.digit_recognizer.seen_digits
        self.areas_to_be_seen_data_label.config(text="\n".join(map(self.digit2gastric, unseen_digits)))
        
        # CHANGE IMAGE
        image_folder = "GastroCheck/GI-Tract-Images/"
        
        # Initialize the background with the default image
        background = Image.open(os.path.join(image_folder, "ProcedureEGD.png")).convert("RGBA")
        
        if seen_areas_text != "":
            numbers_list = sorted(self.digit_recognizer.seen_digits)
            print("NUMBERS")
            print(numbers_list)
            
            # Check if exactly 6 digits are recognized
            if len(numbers_list) == self.num_classes:
                image_name = "ProcedureEGD-all.png"
                background = Image.open(os.path.join(image_folder, image_name)).convert("RGBA")
            else:
                # Loop through recognized digits and overlay images
                for n in numbers_list:
                    image_path = os.path.join(image_folder, f"ProcedureEGD-{n}.png")
                    overlay = Image.open(image_path).convert("RGBA")
                    background = self.combine_images(background, overlay)

        # Resize and update the label with the new image
        self.gi_image = background.resize((400, 600), Image.Resampling.LANCZOS)
        gi_image_tk = ImageTk.PhotoImage(self.gi_image)
        
        # Keep a reference to avoid garbage collection
        self.gi_image.imgtk = gi_image_tk
        self.gi_label.config(image=gi_image_tk)

    def update_detecting_digits_display(self, predicted_class):
        """Updates the label that displays the seen digits."""
        self.current_area_data_label.config(text=self.digit2gastric(predicted_class))
    
    def update_detecting_inside_outside_display(self, predicted_class):
        """Updates the label that displays the seen digits."""
        self.inside_outside_data_label.config(text=self.digit2inside_outside(predicted_class))

    def stop(self):
        """Stops the video feed and releases the camera."""
        self.running = False
        self.cap.release()