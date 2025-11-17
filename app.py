import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load the AI Model ---
# We load this ONCE when the server starts so it's ready immediately
print("Loading AI Model... please wait.")
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'pneumonia_model.h5')

# Check if model exists, otherwise provide helpful error message
if not os.path.exists(MODEL_PATH):
    print("="*60)
    print("WARNING: Model file not found!")
    print(f"Expected location: {MODEL_PATH}")
    print("Please train the model first using: python train_model.py")
    print("The app will continue but predictions will fail.")
    print("="*60)
    model = None
else:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")


def prepare_image(img_path):
    """Preprocesses the image to match the training data format."""
    # 1. Load image and resize to 150x150 (Must match training size!)
    img = image.load_img(img_path, target_size=(150, 150))
    
    # 2. Convert to array (150, 150, 3)
    img_array = image.img_to_array(img)
    
    # 3. Expand dimensions to create a batch (1, 150, 150, 3)
    # The model expects a list of images, even if it's just one.
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Normalize pixel values (0 to 1) (Must match training normalization!)
    img_array /= 255.0
    
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # 1. Save the file locally
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "Please train the model first using: python train_model.py"
            }), 500

        # 2. Preprocess the image
        processed_image = prepare_image(filepath)

        # 3. Make Prediction
        prediction = model.predict(processed_image)
        
        # The model outputs a number between 0 and 1.
        # Close to 0 = NORMAL | Close to 1 = PNEUMONIA
        result_value = prediction[0][0]
        
        if result_value > 0.5:
            result = "Pneumonia Detected"
            # Calculate confidence percentage
            confidence = float(result_value * 100)
        else:
            result = "Normal (Healthy)"
            # Calculate confidence (inverse)
            confidence = float((1 - result_value) * 100)

        return jsonify({
            "result": result, 
            "confidence": f"{confidence:.2f}%", 
            "filename": file.filename
        })


if __name__ == '__main__':
    app.run(debug=True)
