
# Gastro-Check Application

The `Gastro-Check-App` directory contains the core Python scripts for the Gastro-Check project. Below is a description of each script's purpose:

## Script Overview

- **App.py**: The main entry point for the application. This script manages the user interface and integrates video feed capabilities for gastric site recognition.
- **DuplicateItems.py**: Handles the duplication of training data across the parent folder in such a way that every class is represented equally. Essential for ensuring equally distributed training data.
- **Extract_Frames_FolderWise.py**: A utility script to extract video frames and organize them into folders. Useful for preprocessing and dataset preparation.
- **InceptionV3.py**: Implements the InceptionV3 machine learning model for image recognition, particularly trained to detect features in GI tract images.
- **motionMetrics.py**: A specialized script that calculates various motion metrics such as path length, angular length, and average velocity. Essential for evaluating the endoscope movement in GI tract imagery.
- **realTimeGastricSiteRecognition.py**: Handles real-time recognition of gastric sites during analysis. Uses machine learning for accurate gastric site classification.
- **realTimeInsideOutsideRecognition.py**: A module to recognize inside vs. outside GI Tract regions in real time. Crucial for accurately tracking GI tract segments during live video feeds.
- **videoFeed.py**: Manages video input and output streams. This script has been adapted to work seamlessly with inside-outside recognition features.

## How to Use the Scripts

- **App.py**: Use this as the primary script to run the GUI and start the video analysis process.
- **Data Preparation**: Use `Extract_Frames_FolderWise.py` to break down video files into frame images for training and testing purposes. Use `DuplicateItems.py` to duplicate the frame images equally across all classes.
- **Model Training**: Use `InceptionV3.py` to train your model or improve upon existing datasets.
- **Metric Analysis**: `motionMetrics.py` to extract detailed motion data from GI imagery.
- **Real-Time Analysis**: `realTimeDigitRecognition.py` and `realTimeInsideOutsideRecognition.py` for live tracking and analysis tasks.

