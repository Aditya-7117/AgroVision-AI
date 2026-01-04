import numpy as np
from PIL import Image

IMG_SIZE = 224

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
