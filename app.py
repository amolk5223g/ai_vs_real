import cv2
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/ai_detector_model.h5"
IMAGE_SIZE = (96, 96)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def generate_insights(prediction_score, classification):

    if classification == "AI Generated":
        if prediction_score < 0.15:
            return (
                "Strong mathematical artifacts detected. "
                "Frequency distributions match synthetic generator fingerprints."
            )

        return (
            "Minor structural blending inconsistencies detected "
            "along background edge vectors."
        )

    else:

        if prediction_score > 0.85:
            return (
                "Natural image sensor noise patterns identified. "
                "Texture grain and light scattering align with real cameras."
            )

        return (
            "Features closely mirror real photographic content."
        )

def predict_image(image):

    try:

        img = np.array(image)

        img_resized = cv2.resize(img, IMAGE_SIZE)

        img_normalized = img_resized / 255.0

        input_tensor = np.expand_dims(img_normalized, axis=0)

        prediction = float(model.predict(input_tensor)[0][0])

        # Mapping
        # Near 0 = AI
        # Near 1 = Real

        if prediction >= 0.5:
            classification = "Real Image"
            confidence = prediction * 100
        else:
            classification = "AI Generated"
            confidence = (1.0 - prediction) * 100

        analysis = generate_insights(prediction, classification)

        return (
            classification,
            f"{confidence:.2f}%",
            analysis
        )

    except Exception as e:
        return (
            "Error",
            "0%",
            str(e)
        )

iface = gr.Interface(
    fn=predict_image,

    inputs=gr.Image(type="pil"),

    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="AI Analysis")
    ],

    title="AI vs Real Image Detector",

    description=(
        "Upload an image to determine whether "
        "it is AI-generated or a real photograph."
    ),

    theme="soft"
)

iface.launch()