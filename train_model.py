# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("TensorFlow Version:", tf.__version__)

# --- 1. Define Constants ---
# We'll resize all images to 150x150 pixels
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32  # How many images to process at a time
EPOCHS = 10      # How many times to go over the entire dataset

# Define the paths to your data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# --- 2. Prepare the Data (Data Augmentation) ---

# This is a key step. We 'augment' the training images by randomly
# flipping, rotating, and zooming them. This makes our model
# more robust and prevents it from just "memorizing" the images.

train_datagen = ImageDataGenerator(
    rescale=1./255,         # Rescale pixel values from 0-255 to 0-1
    shear_range=0.2,        # Randomly 'shear' the image
    zoom_range=0.2,         # Randomly zoom in
    horizontal_flip=True    # Randomly flip horizontally
)

# For the test data, we ONLY rescale it. We don't augment.
test_datagen = ImageDataGenerator(rescale=1./255)

# --- 3. Create Data Generators ---

# These 'generators' will feed images from the folders to the model
# batch by batch.

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT), # Resize images
    batch_size=BATCH_SIZE,
    class_mode='binary'  # We have 2 classes: NORMAL, PNEUMONIA
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Check the class indices
print("Class Indices:", train_generator.class_indices)

# --- 4. Build the CNN Model ---

model = Sequential([
    # 1st Convolutional Layer
    # 32 filters, 3x3 kernel size, 'relu' activation
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),

    # 2nd Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # 3rd Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten the 3D feature maps into a 1D vector
    Flatten(),

    # A 'Dense' (fully-connected) layer with 512 neurons
    Dense(512, activation='relu'),
    
    # Dropout layer to prevent overfitting (randomly "drops" 50% of neurons)
    Dropout(0.5),

    # Output Layer
    # 1 neuron with 'sigmoid' activation.
    # Sigmoid gives a value between 0 (for 'NORMAL') and 1 (for 'PNEUMONIA')
    Dense(1, activation='sigmoid')
])

# --- 5. Compile the Model ---

model.compile(
    loss='binary_crossentropy', # Good for 2-class (binary) problems
    optimizer='adam',           # A popular and effective optimizer
    metrics=['accuracy']        # We want to see the accuracy
)

# Print a summary of the model's architecture
model.summary()

# --- 6. Train the Model ---

print("Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # Batches per epoch
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE # Batches for validation
)

# --- 7. Save the Trained Model ---

# Ensure the 'model' directory exists
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# Save the model to the 'model' folder
model.save(os.path.join(MODEL_DIR, 'pneumonia_model.h5'))

print("Training complete!")
print("Model saved as 'model/pneumonia_model.h5'")
