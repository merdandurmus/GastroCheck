import cv2
import os

# Paths
video_folder = 'Video'
output_folder = 'Training_Images'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all subfolders in the video folder
for folder_name in os.listdir(video_folder):
    folder_path = os.path.join(video_folder, folder_name)
    
    if os.path.isdir(folder_path):  # Ensure it's a folder
        # Iterate over all video files in the subfolder
        for video_file in os.listdir(folder_path):
            if video_file.endswith('.mp4'):  # Check if the file is a video
                video_name = os.path.splitext(video_file)[0]
                
                # Use the folder name as the output folder in Training_Images
                video_output_path = os.path.join(output_folder, folder_name)
                if not os.path.exists(video_output_path):
                    os.makedirs(video_output_path)

                # Capture the video
                video_path = os.path.join(folder_path, video_file)
                cap = cv2.VideoCapture(video_path)
                
                frame_count = 0
                success = True
                
                # Loop through the frames of the video
                while success:
                    success, frame = cap.read()
                    if success:
                        # Save the current frame in the appropriate folder
                        frame_filename = os.path.join(video_output_path, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(frame_filename, frame)
                        frame_count += 1
                
                cap.release()

                # Print the number of frames saved for each video
                print(f"Saved {frame_count} frames for video '{video_name}' in folder '{video_output_path}'.")

print("Frames extracted and saved successfully.")
