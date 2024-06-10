from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('../../image-classifier/model.h5')  # Load your pre-trained model

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure class names match your model's training classes
class_names = ['000_airplane',
               '001_automobile',
               '002_bird',
               '003_cat',
               '004_deer',
               '005_dog',
               '006_frog',
               '007_horse',
               '008_ship',
               '009_truck'
               ]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))  # Load and resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input
    img_array = img_array / 255.0  # Normalize image to [0, 1] to match the training preprocessing
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    class_name = None
    filename = None
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_name = class_names[predicted_class]
    return render_template('index.html', class_name=class_name, filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
