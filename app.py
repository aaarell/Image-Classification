from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import time

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
model = load_model("mobilenet_model.h5")

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    start_time = time.time()

    img_array = preprocess_image(image_path)
    result = model.predict(img_array)

    end_time = time.time()
    prediction_time = end_time - start_time

    label = np.argmax(result)
    classes = ['Paper', 'Rock', 'Scissors']
    prediction = classes[label]
    accuracy = result[0][label]

    return prediction, accuracy, prediction_time

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        image_filename = file.filename
        image_path = os.path.join('static', image_filename)

        file.save(image_path)
        prediction, accuracy, prediction_time = predict_image(image_path)

        # Get current timestamp
        current_time = int(time.time())

        # Dictionary untuk memetakan label prediksi ke label gambar yang sesuai
        label_mapping = {'Paper': 'paper', 'Rock': 'rock', 'Scissors': 'scissors'}
        prediction_label = label_mapping.get(prediction, 'Unknown')

        return render_template('index.html', prediction=prediction_label, accuracy=accuracy,
                               prediction_time=prediction_time, image_filename=image_filename, current_time=current_time)

if __name__ == '__main__':
    app.run(debug=True)