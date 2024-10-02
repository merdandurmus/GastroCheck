import logging
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard # type: ignore
from tensorflow.keras.layers import BatchNormalization # type: ignore
from argparse import ArgumentParser
from tensorflow.keras.applications import InceptionV3 

# Constants
DEFAULT_IMG_SIZE = (200, 200)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
DEFAULT_VALID_SPLIT = 0.2

# Set up logging
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--imagesize", "-i", help="Image Size", default=DEFAULT_IMG_SIZE)
    parser.add_argument("--trainingdir", "-t", help="Location of training data", default="Data/Training/Training_Images_Colour")
    parser.add_argument("--modelname", "-m", help="Model name", default="Model_Training_Images_Colour")
    parser.add_argument("--gpunumber", "-g", help="GPU number", default="0")
    parser.add_argument("--batchsize", "-b", help="Batch size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--epochs", "-e", help="Number of epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--learningrate", "-lr", help="Learning rate", default=DEFAULT_LR, type=float)
    args = parser.parse_args()
    
    # Validate image size format
    try:
        img_size = tuple(map(int, args.imagesize.split("x")))
        if len(img_size) != 2:
            raise ValueError
    except ValueError:
        raise ValueError("Image size must be in the format WIDTHxHEIGHT (e.g., 200x200)")
    
    return args, (img_size[0], img_size[1], 3)

def configure_gpu(gpu_number):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

def get_data_generators(img_size, dataset_path, batch_size):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
    )
    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size[0], img_size[1]),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size[0], img_size[1]),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    num_classes = train_generator.num_classes
    print(f"Number of classes detected: {num_classes}")
    return train_generator, validation_generator, num_classes

def build_model(img_size, num_classes):
    base_model = InceptionV3(input_shape = img_size, include_top=False, weights = 'imagenet')
    # Unfreeze the top layers of the base model after initial training
    base_model.trainable = True
    for layer in base_model.layers[:-50]:  # Freeze all layers except the last 50: unfreeze a few of the top layers of the base model to allow for fine-tuning.
        layer.trainable = False
    
    # Add custom classification layers
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())  # Replace Flatten with GlobalAveragePooling for better performance
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    
    return model

def train_model(model, train_generator, validation_generator, epochs, steps_per_epoch, validation_steps, callbacks):
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    return history

def save_model(model, model_dir, model_name):
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, model_name))
    print(f"Model saved to {os.path.join(model_dir, model_name)}")

if __name__ == "__main__":
    args, img_size = parse_arguments()
    configure_gpu(args.gpunumber)
    set_random_seeds()
    model_name = f'INCEPTION_{args.modelname}_{args.imagesize}.h5'
    dataset_path = args.trainingdir
    batch_size = args.batchsize
    epochs = args.epochs
    learning_rate = args.learningrate
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"The dataset directory {dataset_path} does not exist.")
    train_generator, validation_generator, num_classes = get_data_generators(img_size, dataset_path, batch_size)
    model = build_model(img_size, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    model_dir = 'Data/models'
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_dir, model_name),
        monitor='accuracy',
        save_best_only=True,
        verbose=1
    )
    #tensorboard = TensorBoard(log_dir='logs')
    callbacks = [early_stopping, reduce_lr, checkpoint]# , tensorboard]
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    history = train_model(model, train_generator, validation_generator, epochs, steps_per_epoch, validation_steps, callbacks)
    #save_model(model, model_dir, model_name)
