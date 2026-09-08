from flask import Flask, request, render_template, redirect
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model("thyroid_model.keras")

# Class names used by the trained model
class_names = ['Benign', 'Malignant', 'Normal']


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

    try:
        # Make the uploaded filename safe
        filename = secure_filename(file.filename)

        # Save uploaded image
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Open and preprocess image
        img = Image.open(filepath).convert("RGB")
        img = img.resize((128, 128))

        # Convert image to NumPy array
        img_array = np.array(img) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Get prediction from trained model
        prediction = model.predict(img_array, verbose=0)[0]

        # Find predicted class
        class_index = int(np.argmax(prediction))
        predicted_class = class_names[class_index]

        # Get model confidence
        confidence = float(prediction[class_index]) * 100

        # Image path for result page
        image_path = f"/static/uploads/{filename}"

        # Render result page
        return render_template(
            "result.html",
            prediction=predicted_class,
            confidence=round(confidence, 2),
            image_path=image_path
        )

    except Exception as e:
        return f"Error during prediction: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)