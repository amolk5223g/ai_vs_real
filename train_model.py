import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import kagglehub

# 1. Fetch dataset path via kagglehub
print("Fetching cached dataset path...")
DOWNLOADED_PATH = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
DATASET_DIR = os.path.join(DOWNLOADED_PATH, "train") 

# Configurations for Stable, High-Accuracy CPU Training
IMAGE_SIZE = (96, 96)       # MobileNetV2 standard input size
BATCH_SIZE = 64
EPOCHS = 3                  # 3 epochs is the sweet spot for a fast, smart CPU train
MODEL_SAVE_PATH = "model/ai_detector_model.h5"

def build_transfer_learning_model():
    # Load Google's pre-trained MobileNetV2 (trained on millions of real-world images)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # Freeze the base layers so your CPU doesn't have to calculate them
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid') # 0 = FAKE (AI), 1 = REAL
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train():
    # Simple rescaling generator
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    print("Setting up data streaming matrices...")
    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    # Verify mapping explicitly: 'FAKE' folder = 0, 'REAL' folder = 1
    print("Dataset Class Mapping:", train_generator.class_indices)
    
    model = build_transfer_learning_model()
    
    print("\n[+] Training Advanced MobileNetV2 Transfer Learning Model...")
    # 200 steps checks 12,800 images—perfect for a smart, diverse feature set
    model.fit(
        train_generator,
        steps_per_epoch=200,       
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=40        
    )
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n[+] Success! Smart model written to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()