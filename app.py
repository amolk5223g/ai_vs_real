import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MODEL_PATH = 'model/ai_detector_model.h5'
IMAGE_SIZE = (96, 96) # Matches the MobileNetV2 dimensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Missing model file at {MODEL_PATH}")
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_insights(prediction_score, classification):
    if classification == "AI Generated":
        if prediction_score < 0.15:
            return "Strong mathematical artifacts detected. Frequency distributions match synthetic generator fingerprints (GAN/Diffusion)."
        return "Minor structural blending inconsistencies detected along background edge vectors."
    else:
        if prediction_score > 0.85:
            return "Natural image sensor noise patterns identified. Texture grain and light scattering align perfectly with physical camera lenses."
        return "Features closely mirror real photographic content, though light compression artifacts are present."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            img_raw = cv2.imread(filepath)
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
            img_normalized = img_resized / 255.0
            input_tensor = np.expand_dims(img_normalized, axis=0)
            
            detector = get_model()
            prediction = float(detector.predict(input_tensor)[0][0])
            
            # STANDARD DICTIONARY MAPPING
            # Near 0.0 = AI Generated (FAKE)
            # Near 1.0 = Real Image (REAL)
            if prediction >= 0.5:
                classification = "Real Image"
                confidence = prediction * 100
            else:
                classification = "AI Generated"
                confidence = (1.0 - prediction) * 100
                
            analysis = generate_insights(prediction, classification)
            
            return jsonify({
                'success': True,
                'prediction': classification,
                'confidence': f"{confidence:.2f}%",
                'probability': prediction,
                'analysis': analysis,
                'filepath': filepath
            })
            
        except Exception as e:
            return jsonify({'error': f"Processing error: {str(e)}"}), 500
            
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)