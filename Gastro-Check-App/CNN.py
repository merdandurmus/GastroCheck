import os
import random
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, model_name, output_dir='Data/confusion_matrices'):
        self.validation_data = validation_data
        self.model_name = model_name
        self.output_dir = os.path.join(output_dir, model_name)  # Create subfolder for model

        # Ensure the directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_labels = self.validation_data
        predicted_labels = np.argmax(self.model.predict(val_images), axis=1)
        true_labels = np.argmax(val_labels, axis=1)

        # Compute the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(self.model.output_shape[-1]), yticklabels=range(self.model.output_shape[-1]))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix for model {self.model_name} - Epoch {epoch + 1}')
        
        save_path = os.path.join(self.output_dir, f'confusion_matrix_model_{self.model_name}_epoch_{epoch + 1}.png')
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory

class CustomDigitClassifier:
    def __init__(self, dataset_path, num_classes, img_size, model_name):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.model_name = model_name

    def visualize_samples_paginated(self, images, labels, labelshift, samples_per_page=25):
        """
        Visualize samples in a paginated way.
        Args:
        - images: Array of images (num_samples, img_size[0], img_size[1], 1)
        - labels: One-hot encoded labels corresponding to the images
        - labelshift: Boolean indicating whether labels have been shifted
        - img_size: Size of each image (height, width)
        - samples_per_page: Number of samples to visualize per page
        """
        # Ensure that `images` is a NumPy array
        if not isinstance(images, np.ndarray):
            raise TypeError(f"Expected images to be a NumPy array, got {type(images)} instead.")

        total_samples = images.shape[0]

        # Randomly shuffle the images and labels
        indices = random.sample(range(total_samples), total_samples)
        images = images[indices]
        labels = labels[indices]

        # Total number of pages
        total_pages = (total_samples + samples_per_page - 1) // samples_per_page

        def plot_page(page_idx):
            """ Helper function to plot a specific page. """
            start_idx = page_idx * samples_per_page
            end_idx = min(start_idx + samples_per_page, total_samples)
            n_samples = end_idx - start_idx

            n_cols = 5
            n_rows = (n_samples + n_cols - 1) // n_cols

            fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
            axs = axs.flatten()

            for i in range(n_samples):
                idx = start_idx + i
                img = images[idx]  # Reshape back to 2D for plotting
                if labelshift:
                    label = np.argmax(labels[idx]) - 1  # Subtract 1 to reverse the earlier label shift
                else:
                    label = np.argmax(labels[idx])  # Do not subtract 1 since no label shift
                
                axs[i].imshow(img, cmap='gray')
                axs[i].set_title(f'Label: {label}', fontsize=10)
                axs[i].axis('off')  # Remove axes for a cleaner look

            # Turn off unused subplots
            for j in range(n_samples, len(axs)):
                axs[j].axis('off')

            plt.tight_layout()
            plt.show()

        # Function to navigate through pages
        current_page = 0

        def scroll_pages(event):
            nonlocal current_page
            if event.key == 'ArrowRight':
                current_page = (current_page + 1) % total_pages  # Scroll forward
            elif event.key == 'ArrowLeft':
                current_page = (current_page - 1) % total_pages  # Scroll backward
            plt.close()  # Close the current figure
            plot_page(current_page)

        # Initial plot
        plot_page(current_page)

        # Connect keyboard events to scroll
        fig = plt.gcf()  # Get current figure
        fig.canvas.mpl_connect('key_press_event', scroll_pages)

    def load_custom_dataset(self, labelshift):
        images = []
        labels = []

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest'
        )

        # Iterate through each folder (assuming each folder name is the digit label)
        for label in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label)
            print(f"Loading dataset: {label_path}" " ...")
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)

                    # Check if the file is a valid image before processing
                    if img_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Add image extensions
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        
                        # Ensure the image was loaded properly
                        if img is not None:
                            img_resized = cv2.resize(img, (self.img_size[0], self.img_size[1]))
                            images.append(img_resized)
                            if labelshift:
                                labels.append(int(label) + 1)  # Shift labels since -1 class is used
                            else:
                                labels.append(int(label)) # No shift since there is no -1 class
                        else:
                            print(f"Warning: Failed to load image {img_path}")
                    else:
                        print(f"Skipping non-image file: {img_name}")
            print(f"Done!")

        # Convert lists to numpy arrays
        images = np.array(images).reshape(-1, self.img_size[0], self.img_size[1], self.img_size[2]).astype('float32') / 255
        labels = np.array(labels)

        print(f"Hot-Encoding Labels...")
        # One-hot encode the labels
        labels = to_categorical(labels, num_classes=self.num_classes)

        print(f"Augementing images...")
        # Augment images
        images_augmented = []
        labels_augmented = []
        for img, lbl in zip(images, labels):
            img = img.reshape((1,) + img.shape)
            for batch in datagen.flow(img, batch_size=1):
                images_augmented.append(batch[0])
                labels_augmented.append(lbl)  # Append the correct label
                break
        
        print(f"Successfully loaded & augmented dataset: {self.dataset_path}")
        return np.array(images_augmented), np.array(labels_augmented)

    def build_model(self):
        print(f"Building model: {self.model_name}" "...")
        model = models.Sequential()

        # First convolutional block
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_regularizer=regularizers.l2(0.001),
                                input_shape=(self.img_size[0], self.img_size[1], 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Second convolutional block
        model.add(layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Third convolutional block
        model.add(layers.Conv2D(256, (3, 3), activation='relu',
                                kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Flatten and dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        print(f"Model built!")
    
    def train_model(self, train_images, train_labels, test_images, test_labels, epochs=13, batch_size=64):
        """Train the CNN model."""
        if self.model is None:
            self.build_model()
        
        cm_callback = ConfusionMatrixCallback(validation_data=(test_images, test_labels), model_name=self.model_name)
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        early = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, 
                      verbose=1, mode="min", baseline=None, restore_best_weights=False)

        # Calculate class weights using original training labels before augmentation
        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(np.argmax(train_labels, axis=1)),
                                                          y=np.argmax(train_labels, axis=1))

        # Convert to dictionary format
        class_weight_dict = dict(enumerate(class_weights))

        print("Starting training...")
        history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels), class_weight=class_weight_dict, callbacks=[cm_callback, early_stopping, early])
        print("Training completed.")

        """Save the CNN model."""
        model_dir = 'Data/models'
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, self.model_name))
        print(f"Model saved to {os.path.join(model_dir, self.model_name)}")

        return history

    def evaluate_model(self, test_images, test_labels):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Evaluate the model on the test set
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print(f"Test accuracy: {test_acc:.4f}")
        return test_acc

# Usage:
# Initialize the classifier with dataset path
model_name = 'colour.h5' # CHANGE!!!!!!!!!!!!!!
dataset_path='Data/Training/Training_Images_Colour' # CHANGE!!!!!!!!!!!!!!
labelshift=False # CHANGE!!!!!!!!!!!!!!
classifier = CustomDigitClassifier(model_name=model_name, dataset_path=dataset_path, num_classes=4, img_size=(50, 50, 3)) # CHANGE!!!!!!!!!!!!!!

# Load dataset
images, labels = classifier.load_custom_dataset(labelshift)

# Plot the label distribution before training
labels_numeric = np.argmax(labels, axis=1)
plt.hist(labels_numeric, bins=np.arange(7) - 0.5, rwidth=0.8)
plt.xticks(range(6))
plt.xlabel('Digit Class')
plt.ylabel('Frequency')
plt.show()

classifier.visualize_samples_paginated(images, labels, labelshift)

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the model
classifier.train_model(train_images, train_labels, test_images, test_labels)

# Evaluate the model on test data
classifier.evaluate_model(test_images, test_labels)
