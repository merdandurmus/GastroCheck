import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore


# Define the path to the test directory
test_dir = "GastroCheck/Data/TEST/TESTColours_Pattern_8sites"  # Replace with your actual test directory path
image_size = (100, 100)  # The image size used for your model (as per your model)
batch_size = 32  # Batch size, you can adjust it based on memory

# Load the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=image_size[:2],  # (height, width)
    batch_size=batch_size,
    label_mode='categorical',  # Assumes the directory names are integers (0, 1, etc.)
    shuffle=False  # No need to shuffle when evaluating
)

# Load your trained model
model = keras.models.load_model("GastroCheck/Data/models/INCEPTIONV3_Colours-Patterns-8Sites_100x100.h5")

# Compile the model with the correct metrics
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Ensure the loss matches your label encoding
    metrics=['accuracy']  # Use a flat list for metrics
)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

