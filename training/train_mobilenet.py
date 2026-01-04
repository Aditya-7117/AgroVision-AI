import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os

# =====================
# CONFIG
# =====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_DIR = "backend/model"

os.makedirs(MODEL_DIR, exist_ok=True)

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
# MODEL
# =====================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================
# CALLBACKS
# =====================
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "veg_disease_mobilenet.h5"),
        save_best_only=True
    )
]

# =====================
# TRAIN
# =====================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================
# SAVE CLASS LABELS
# =====================
with open(os.path.join(MODEL_DIR, "class_labels.json"), "w") as f:
    json.dump(train_data.class_indices, f, indent=4)

print("Training completed. Model and labels saved.")
