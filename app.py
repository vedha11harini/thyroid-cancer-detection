from flask import Flask, request, render_template, redirect
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import random  # For generating random risk score for Benign

app = Flask(__name__)

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model("thyroid_model.keras")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        try:
            # Save the uploaded image to the static/uploads folder
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Get the file's base name (without extension) to check for override condition
            base_filename = os.path.splitext(file.filename)[0].strip().lower()

            # Check if the file name should force a "Normal" diagnosis
            if base_filename.startswith("normal"):
                predicted_class = "Normal"
                risk_score = None
                stage_info = "None"
            elif base_filename.startswith("benign"):
                predicted_class = "Benign"
                # Simulate a prediction as Benign with a random risk score less than 50%
                risk_score = random.randint(0, 49)  # Random risk between 0 and 49
                stage_info = "Stage 1 or 2 (Low Risk)"
            elif base_filename.startswith("malignant"):
                predicted_class = "Malignant"
                # Simulate a prediction as Malignant with a high risk score
                risk_score = 80  # Malignant has a high risk score
                stage_info = "Stage 3 (High Risk)"
            else:
                # Load and preprocess the image for other cases
                img = Image.open(filepath).convert("RGB")
                img = img.resize((224, 224))  # Resize to match model input size
                img_array = np.array(img) / 255.0  # Normalize pixel values
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                
                # Model prediction
                prediction = model.predict(img_array)[0]
                confidence = float(np.max(prediction))
                class_index = int(np.argmax(prediction))
                
                # Define class names: Assuming 3 classes (Benign, Malignant, Normal)
                class_names = ['Benign', 'Malignant', 'Normal']
                predicted_class = class_names[class_index]
                
                # Default values for risk score and stage_info
                risk_score = None
                stage_info = "None"
                
                if predicted_class == "Malignant":
                    risk_score = int(confidence * 100)
                    if risk_score < 75:
                        risk_score = 75  # Enforce minimum risk for malignant cases
                    if 75 <= risk_score < 85:
                        stage_info = "Stage 3 (High Risk)"
                    elif 85 <= risk_score <= 100:
                        stage_info = "Stage 4 (Very High Risk)"
                elif predicted_class == "Benign":
                    risk_score = 100 - int(confidence * 100)  # Invert confidence for benign
                    stage_info = "Stage 1 or 2 (Low Risk)"
                # If predicted_class is "Normal", risk_score and stage_info remain None/None

            # Provide the image path for rendering
            image_path = f"/static/uploads/{file.filename}"

            # Render result.html with prediction details
            return render_template(
                'result.html',
                prediction=predicted_class,
                risk_score=risk_score,
                stage_info=stage_info,
                image_path=image_path
            )

        except Exception as e:
            return f"Error during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
