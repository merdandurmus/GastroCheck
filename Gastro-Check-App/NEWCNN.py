import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--imagesize", "-i", help="Image Size", default="200x200")
    parser.add_argument("--trainingdir", "-t", help="Location of training data", default="Data/Training/Training_Images_Colour")
    parser.add_argument("--modelname", "-m", help="Model name", default="Model_Training_Images_Colour")
    parser.add_argument("--gpunumber", "-g", help="Model name", default="0")
    args = parser.parse_args()
    
    # Define GPU visible devices (env var)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpunumber # String value = 0, 1, 2
    
    sizes = args.imagesize.split("x")
    
    # Usage:
    # Initialize the classifier with dataset path
    img_size=(int(sizes[0]), int(sizes[1]), 3)
    model_name = f'{args.modelname}_{args.imagesize}.h5'
    dataset_path=args.trainingdir
    labelshift=False # CHANGE!!!!!!!!!!!!!!


    # Create a data generator with a validation split (e.g., 20% for validation)
    datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize the images
        validation_split=0.2,     # Use 20% of the data for validation
        horizontal_flip=False,
        fill_mode='nearest',
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2
    )

    # Training data generator (80% of the data)
    train_generator = datagen.flow_from_directory(
        dataset_path,       # Main dataset directory
        target_size=(img_size[0], img_size[1]),
        batch_size=32,
        class_mode='categorical',
        subset='training'        # Specify this is the training subset
    )

    # Validation data generator (20% of the data)
    validation_generator = datagen.flow_from_directory(
        dataset_path,       # Main dataset directory
        target_size=(img_size[0], img_size[1]),
        batch_size=32,
        class_mode='categorical',
        subset='validation'      # Specify this is the validation subset
    )
    
    # Dynamically set the number of classes based on the training data
    num_classes = train_generator.num_classes
    print(f"Number of classes detected: {num_classes}")
    
    print(f"Building model: ...")
    model = models.Sequential()
    # First convolutional block
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001),
                            input_shape=(img_size[0], img_size[1], 3)))
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

    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Model built!")
    
    

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    # Fit the model using the generators
    model.fit(
        train_generator,
        epochs=50,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping]
    )

    # After training, evaluate the model using the test data ! ADD SEPARATE TEST SET
    # test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
    # print(f'Test accuracy: {test_acc}')

    """Save the CNN model."""
    model_dir = 'Data/models'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, model_name))
    print(f"Model saved to {os.path.join(model_dir, model_name)}")