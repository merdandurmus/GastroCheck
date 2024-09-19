import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

class CustomDigitClassifier:
    def __init__(self, dataset_path, img_size=(28, 28), num_classes=6):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
    
    def load_custom_dataset(self):
        images = []
        labels = []

        # Iterate through each folder (assuming each folder name is the digit label)
        for label in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)

                    # Check if the file is a valid image before processing
                    if img_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Add image extensions
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        # Ensure the image was loaded properly
                        if img is not None:
                            img_resized = cv2.resize(img, self.img_size)
                            images.append(img_resized)
                            labels.append(int(label) )# + 1)  # The folder name is the label
                        else:
                            print(f"Warning: Failed to load image {img_path}")
                    else:
                        print(f"Skipping non-image file: {img_name}")

        # Convert lists to numpy arrays
        images = np.array(images).reshape(-1, self.img_size[0], self.img_size[1], 1).astype('float32') / 255
        labels = np.array(labels)

        # One-hot encode the labels
        labels = to_categorical(labels, num_classes=self.num_classes)

        return images, labels

    def build_model(self):
        # Define the CNN model architecture
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    
    def train_model(self, train_images, train_labels, test_images, test_labels, epochs=9, batch_size=64):
        """Train the CNN model."""
        if self.model is None:
            self.build_model()
        print("Starting training...")
        history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels))
        print("Training completed.")

        """Save the CNN model."""
        model_dir = 'models'
        model_name = 'Digit_Classifier_Gastro_Images_MNIST.h5'
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, model_name))
        print(f"Model saved to {os.path.join(model_dir, model_name)}")

        
        return history

    def evaluate_model(self, test_images, test_labels):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Evaluate the model on the test set
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print(f"Test accuracy: {test_acc:.4f}")
        return test_acc

# Example usage:

# Initialize the classifier with dataset path
classifier = CustomDigitClassifier(dataset_path='Training_Images_MNIST')

# Load dataset
images, labels = classifier.load_custom_dataset()

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the model
classifier.train_model(train_images, train_labels, test_images, test_labels)

# Evaluate the model on test data
classifier.evaluate_model(test_images, test_labels)
