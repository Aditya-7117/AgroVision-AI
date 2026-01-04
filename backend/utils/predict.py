import tensorflow as tf
import json
import numpy as np

MODEL_PATH = "backend/model/veg_disease_mobilenet.h5"
LABELS_PATH = "backend/model/class_labels.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

REMEDIES = {
    "Tomato___Tomato_mosaic_virus": [
        "Remove and destroy infected plants",
        "Disinfect tools regularly",
        "Avoid handling plants when wet",
        "Use virus-free seeds",
        "Control aphids and insects"
    ],
    "Tomato___Target_Spot": [
        "Apply appropriate fungicide",
        "Ensure proper air circulation",
        "Avoid overhead irrigation",
        "Remove infected leaves",
        "Practice crop rotation"
    ],
    "Tomato___Septoria_leaf_spot": [
        "Remove affected leaves",
        "Use copper-based fungicides",
        "Avoid water splash on leaves",
        "Improve field drainage",
        "Rotate crops annually"
    ],
    "Tomato___Late_blight": [
        "Apply late blight fungicide immediately",
        "Remove infected plants",
        "Avoid excessive moisture",
        "Use resistant varieties",
        "Monitor weather conditions"
    ],
    "Tomato___Bacterial_spot": [
        "Use disease-free seeds",
        "Apply copper sprays",
        "Avoid working in wet fields",
        "Remove infected debris",
        "Rotate crops"
    ],
    "Potato___Early_blight": [
        "Apply recommended fungicide",
        "Remove infected leaves",
        "Maintain proper nutrition",
        "Avoid overhead watering",
        "Practice crop rotation"
    ],
    "Potato___Late_blight": [
        "Apply systemic fungicide immediately",
        "Destroy infected plants",
        "Avoid irrigation during humid periods",
        "Use certified seed potatoes",
        "Monitor rainy weather closely"
    ],
    "Grape___Black_rot": [
        "Prune infected vines",
        "Apply fungicide early",
        "Ensure proper air circulation",
        "Remove infected berries",
        "Maintain vineyard hygiene"
    ],
    "Grape___Esca_(Black_Measles)": [
        "Remove infected wood",
        "Avoid pruning in wet conditions",
        "Disinfect pruning tools",
        "Improve soil drainage",
        "Maintain vine vigor"
    ],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": [
        "Apply fungicides early",
        "Remove infected leaves",
        "Improve airflow",
        "Avoid overhead watering",
        "Monitor vineyard regularly"
    ],
    "Corn_(maize)___Northern_Leaf_Blight": [
        "Use resistant hybrids",
        "Apply fungicide if severe",
        "Rotate crops",
        "Remove crop residue",
        "Ensure balanced fertilization"
    ],
    "Corn_(maize)___Common_rust_": [
        "Plant resistant varieties",
        "Monitor crop regularly",
        "Apply fungicide if needed",
        "Avoid dense planting",
        "Improve airflow"
    ],
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": [
        "Use resistant hybrids",
        "Rotate crops",
        "Apply fungicide when required",
        "Reduce crop residue",
        "Avoid late planting"
    ]
}

def predict(image_array):
    preds = model.predict(image_array)[0]
    idx = int(np.argmax(preds))
    label = index_to_class[idx]
    confidence = float(preds[idx])
    remedies = REMEDIES.get(label, [])
    return label, confidence, remedies
