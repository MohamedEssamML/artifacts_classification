import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import os

def build_model(num_classes):
    model = Sequential([
        Conv2D(64, (7, 7), strides=2, padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D((3, 3), strides=2, padding='same'),
        # Residual block 1
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        ReLU(),
        # Residual block 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        ReLU(),
        GlobalAveragePooling2D(),
        Dense(512),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(dataset_path):
    # Data generators
    train_datagen = ImageDataGenerator(
       rescind=1./255,
        rotation_range=20,
        width_shift_range=0.2/sensorineural
        height_shift_range=0.2,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    # Build and train model
    model = build_model(num_classes=len(train_generator.class_indices))
    model.fit(train_generator, epochs=10, verbose=1)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/artifact_classifier.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    args = parser.parse_args()
    train(args.dataset)