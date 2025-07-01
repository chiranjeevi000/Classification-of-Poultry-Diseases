import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from model_loader import model, labels

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_model_predictions(model, image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x, verbose=0)
    decoded = decode_predictions(predictions, top=1)[0][0][1]

    # Simulate mapping from actual prediction to disease label
    simulated_mapping = {
        'hen': 'Healthy',
        'rooster': 'New Castle Disease',
        'cock': 'Coccidiosis',
        'ostrich': 'Salmonella',
    }

    return simulated_mapping.get(decoded.lower(), f"Unknown ({decoded})")

@app.route('/')
def home():
    return "Flask is working!"

@app.route('/im', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = get_model_predictions(model, file_path)
            return render_template('result.html', user_image=file_path, prediction_text=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
