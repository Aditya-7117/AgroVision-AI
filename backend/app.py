from flask import Flask, render_template, request, jsonify
from utils.preprocess import preprocess_image
from utils.predict import predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files["image"]
    image = preprocess_image(file)

    label, confidence, remedies = predict(image)

    return jsonify({
        "disease": label.replace("___", " - "),
        "confidence": round(confidence * 100, 2),
        "remedies": remedies
    })

if __name__ == "__main__":
    app.run(debug=True)
