from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("thyroid_model.keras")

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Function to predict image
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize pixels (0-1 scale)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Model prediction
    prediction = model.predict(img_array)[0]  # Get probabilities for each class
    confidence = np.max(prediction)  # Highest probability
    predicted_class = np.argmax(prediction)

    # Calculate risk score (0–100)
    risk_score = int(confidence * 100)

    # Risk-based diagnosis and stage logic
    if risk_score < 50:
        diagnosis = "Normal"
        cancer_stage = "None"
    elif 50 <= risk_score < 70:
        diagnosis = "Benign"
        cancer_stage = "Stage 1 or 2 (Low Risk)"
    elif 70 <= risk_score < 85:
        diagnosis = "Possibly Malignant"
        cancer_stage = "Stage 2 or 3 (Moderate Risk)"
    else:
        diagnosis = "Malignant"
        cancer_stage = "Stage 4 (High Risk)"

    return diagnosis, risk_score, cancer_stage

# Upload page
@app.route("/upload")
def upload_page():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    filepath = "uploaded_image.png"
    file.save(filepath)

    # Predict image
    diagnosis, risk_score, cancer_stage = predict_image(filepath)

    return render_template("index.html", diagnosis=diagnosis, risk_score=risk_score, cancer_stage=cancer_stage)

if __name__ == "__main__":
    app.run(debug=True)
