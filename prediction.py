import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Configuration for the Keras model
MODEL_URL = "https://flowerm.s3.us-east-1.amazonaws.com/flower_model_final.keras"
MODEL_PATH = "flower_model_final.keras"
CLASS_LABELS_PATH = "data/class_labels.json"

# Function to ensure the model exists locally
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"Downloading model from {MODEL_URL}...")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model downloaded successfully to {MODEL_PATH}.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading model: {e}")

# Load the TensorFlow model
def load_model_and_labels():
    # Ensure the model exists before loading
    ensure_model_exists()
    model = load_model(MODEL_PATH)
    # Load class labels
    with open(CLASS_LABELS_PATH, "r") as f:
        class_labels = json.load(f)
    return model, class_labels

# Function to predict the flower
def predict_flower(img, model, class_labels):
    """
    Predict the type of flower from the image.

    Parameters:
        img (PIL.Image): The input image.
        model (Keras.Model): The loaded model.
        class_labels (dict): Mapping of class indices to flower names.

    Returns:
        tuple: Predicted label and confidence percentage.
    """
    img_width, img_height = 288, 276
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    confidence = np.max(predictions) * 100
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class)]

    return (predicted_label, confidence) if confidence >= 80 else (None, None)
