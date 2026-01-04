import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# =====================
# CONFIG
# =====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_PATH = "backend/model/veg_disease_mobilenet.h5"

# =====================
# DATA GENERATORS
# =====================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# =====================
# LOAD MODEL
# =====================
model = load_model(MODEL_PATH)

# =====================
# FINE-TUNING STRATEGY
# =====================
# Freeze everything first
for layer in model.layers:
    layer.trainable = False

# Unfreeze last 30 convolutional layers ONLY
trainable_count = 0
for layer in reversed(model.layers):
    if "conv" in layer.name:
        layer.trainable = True
        trainable_count += 1
    if trainable_count >= 30:
        break

print(f"Unfroze {trainable_count} convolutional layers for fine-tuning.")

# =====================
# RECOMPILE WITH LOW LR
# =====================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================
# CALLBACKS
# =====================
callbacks = [
    EarlyStopping(patience=2, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

# =====================
# FINE-TUNE
# =====================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("Fine-tuning completed successfully.")
