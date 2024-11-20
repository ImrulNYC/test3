import streamlit as st
from PIL import Image
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests  # Ensure this is imported
from tensorflow.keras.models import load_model
import json

# Configuration for the Keras model
MODEL_URL = "https://flowerm.s3.us-east-1.amazonaws.com/flower_model_final.keras"
MODEL_PATH = "flower_model_best.keras"
CLASS_LABELS_PATH = "data/class_labels.json"

# Ensure the model exists locally
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"Downloading model from {MODEL_URL}...")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()  # Raise exception for HTTP errors
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model downloaded successfully to {MODEL_PATH}.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading model: {e}")

# Call the function to ensure the model exists before loading
ensure_model_exists()


# Call the function to ensure the model exists before loading
ensure_model_exists()

# Load the model globally for predictions
model = load_model(MODEL_PATH)

# Function to get current Eastern Time
def get_current_eastern_time():
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern).strftime("%Y-%m-%d %H:%M:%S")

# Streamlit app title and configuration
st.set_page_config(
    page_title="Flower Identification App",
    page_icon="🌼",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Flower Identification App 🌼")
st.markdown("Upload a flower image to discover its name. 🌺")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Developer Info"])

if page == "Home":
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"], 
        help="Upload an image file (JPG, JPEG, PNG) to identify the flower."
    )

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image_obj = Image.open(uploaded_file)
            st.image(image_obj, caption='Uploaded Image', use_column_width=True)

            # Predict flower type
            def predict_flower(img):
                """
                Predict the type of flower from the image.

                Parameters:
                    img: PIL Image object.

                Returns:
                    tuple: Predicted label and confidence percentage.
                """
                img_width, img_height = 288, 276
                img = img.resize((img_width, img_height))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0  # Normalize image

                # Predict using the loaded model
                predictions = model.predict(img_array)
                confidence = np.max(predictions) * 100  # Get confidence in percentage
                predicted_class = np.argmax(predictions, axis=1)[0]

                # Load class labels
                with open(CLASS_LABELS_PATH, "r") as f:
                    class_labels = json.load(f)

                predicted_label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class)]

                # Return the result if confidence is high enough
                if confidence >= 80:
                    return predicted_label, confidence
                else:
                    return None, None

            # Run prediction
            with st.spinner("Identifying the flower..."):
                predicted_label, confidence = predict_flower(image_obj)

            # Display results
            if predicted_label:
                st.success(f"Predicted Flower: **{predicted_label}** with **{confidence:.2f}%** confidence.")
            else:
                st.warning("The flower cannot be confidently recognized. Please try another image.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload an image to get started.")

elif page == "Developer Info":
    # Developer information
    st.header("Developer Team")
    st.markdown(
        """
        **Team Members:**
        - Jessica
        - Mansur
        - Zahava
        - Imrul

        **About the App:**
        - Built using TensorFlow for flower identification.
        - Model: Pretrained Keras model fine-tuned for flower classification.
        """
    )
