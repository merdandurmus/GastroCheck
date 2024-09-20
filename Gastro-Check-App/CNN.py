import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, model_name):
        self.validation_data = validation_data
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_labels = self.validation_data
        predictions = self.model.predict(val_images)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(val_labels, axis=1)

        # Compute the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(self.model.output_shape[-1]), yticklabels=range(self.model.output_shape[-1]))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix for model {self.model_name} - Epoch {epoch + 1}')
        
        save_path = os.path.join(self.output_dir, f'confusion_matrix_epoch_{epoch + 1}.png')
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory

class CustomDigitClassifier:
    def __init__(self, dataset_path, img_size=(200, 200), num_classes=6):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
    
    def load_custom_dataset(self):
        images = []
        labels = []

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

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
                            labels.append(int(label))# + 1)  # The folder name is the label
                        else:
                            print(f"Warning: Failed to load image {img_path}")
                    else:
                        print(f"Skipping non-image file: {img_name}")

        # Convert lists to numpy arrays
        images = np.array(images).reshape(-1, self.img_size[0], self.img_size[1], 1).astype('float32') / 255
        labels = np.array(labels)

        # One-hot encode the labels
        labels = to_categorical(labels, num_classes=self.num_classes)

        # Augment images
        images_augmented = []
        labels_augmented = []
        for img, lbl in zip(images, labels):
            img = img.reshape((1,) + img.shape)
            for batch in datagen.flow(img, batch_size=1):
                images_augmented.append(batch[0])
                labels_augmented.append(lbl)  # Append the correct label
                break

        return np.array(images_augmented), np.array(labels_augmented)

    def build_model(self):
        # Define the CNN model architecture
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))  # Add dropout after first convolutional block

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))  # Add dropout after first convolutional block

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # Increased filter size
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))  # Add dropout after first convolutional block

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))  # Increased dense layer size
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    
    def train_model(self, train_images, train_labels, test_images, test_labels, epochs=15, batch_size=64):
        """Train the CNN model."""
        if self.model is None:
            self.build_model()
        
        model_name = 'Digit_Classifier_Gastro_200x200_Images_without__-1.h5'

        cm_callback = ConfusionMatrixCallback(validation_data=(test_images, test_labels), model_name=model_name)

        # Calculate class weights using original training labels before augmentation
        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(np.argmax(train_labels, axis=1)),
                                                          y=np.argmax(train_labels, axis=1))

        # Convert to dictionary format
        class_weight_dict = dict(enumerate(class_weights))

        print("Starting training...")
        history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels), class_weight=class_weight_dict, callbacks=[cm_callback])
        print("Training completed.")

        """Save the CNN model."""
        model_dir = 'models'
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
classifier = CustomDigitClassifier(dataset_path='Training_Images_without_-1')

# Load dataset
images, labels = classifier.load_custom_dataset()

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the model
classifier.train_model(train_images, train_labels, test_images, test_labels)

# Evaluate the model on test data
classifier.evaluate_model(test_images, test_labels)
