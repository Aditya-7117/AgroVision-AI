# AgroVision AI 🌱  
AI-Powered Vegetable Disease Detection System

## Overview
AgroVision AI is a deep learning–based web application designed to help farmers and agricultural practitioners identify crop diseases from leaf images. The system analyzes an uploaded leaf image and predicts the disease class along with a confidence score and actionable remedial steps. The application is built with a focus on accuracy, efficiency, and real-world usability.

---

## Project Phases & Model Evolution

### Phase 1 – Custom CNN (Initial Approach)
In the initial phase, the project was implemented using a custom Convolutional Neural Network (CNN). While the model achieved reasonable accuracy, it required higher computational resources and longer training times, making it less suitable for low-end or CPU-only systems.

### Phase 2 – Transfer Learning with MobileNetV2 (Final Approach)
To improve efficiency and deployment feasibility, the project was upgraded to **MobileNetV2 using transfer learning**. This significantly reduced training cost and inference time while maintaining strong accuracy.  
This transition allows the model to run smoothly on standard laptops and CPU-based environments, making it practical for academic, demo, and real-world use.

---

## Supported Crops & Diseases

### Tomato
- Bacterial Spot  
- Late Blight  
- Septoria Leaf Spot  
- Target Spot  
- Tomato Mosaic Virus  

### Potato
- Early Blight  
- Late Blight  

### Grape
- Black Rot  
- Esca (Black Measles)  
- Leaf Blight  
- Healthy  

### Corn (Maize)
- Cercospora / Gray Leaf Spot  
- Common Rust  
- Northern Leaf Blight  
- Healthy  

---

## Key Features
- Upload leaf image and get instant disease prediction  
- Confidence score visualized using a donut chart  
- Disease-specific remedial actions for farmers  
- Modern dark-themed UI with stable layout  
- CPU-optimized model for low-cost deployment  

---

## Tech Stack
- **Model**: MobileNetV2 (Transfer Learning)  
- **Framework**: TensorFlow / Keras  
- **Backend**: Flask  
- **Frontend**: HTML, CSS, JavaScript  
- **Dataset**: PlantVillage (Labeled)  

---

## Project Structure
```
AgroVision-AI/
│
├── backend/
│   ├── app.py
│   ├── model/
│   │   ├── veg_disease_mobilenet.h5
│   │   └── class_labels.json
│   ├── utils/
│   │   ├── preprocess.py
│   │   └── predict.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       ├── js/main.js
│       └── assets/bg.jpg
│
├── training/
│   ├── split_dataset.py
│   ├── train_mobilenet.py
│   └── fine_tune_mobilenet.py
│
├── dataset/
│   ├── train/
│   └── val/
│
├── screenshots/
│   ├── homepage.png
│   ├── image_upload.png
│   └── prediction_output.png
│
└── README.md

```
---

## How to Run the Project

1. Create and activate a virtual environment  

2. Install dependencies:
```
pip install -r backend/requirements.txt
```
3. Run the Flask application:
```
python backend/app.py
```

4. Open in browser:
```
http://127.0.0.1:5000
```
---

## Model Training Summary
- Initial training performed with MobileNetV2 backbone frozen  
- Fine-tuning applied to upper convolutional layers  
- Early stopping used to prevent overfitting  
- Achieved approximately 90–92% validation accuracy across 16 disease classes  

---

## Screenshots
- **homepage.png** – Landing page with logo and upload card  
- **image_upload.png** – Leaf image preview after upload  
- **prediction_output.png** – Disease prediction, confidence chart, and remedies  

---

## Disclaimer
This application provides decision support only. Farmers should consult agricultural experts before taking critical actions based on predictions.

---

## Author
Aditya Magar
