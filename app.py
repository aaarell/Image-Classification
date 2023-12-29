import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Gets rid of GPU errors from tensorflow

import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/")
def index():
    return render_template('/select.html', )

@app.route('/predict', methods=['POST'])
def predict():
    chosen_model = request.form['select_model']
    model_dict = {'MobileNet Model'   :   'static/MobileNetrps.h5',
                  'ResNet Model'     :   'static/ResNetrps.h5',}
    if chosen_model in model_dict:
        model = load_model(model_dict[chosen_model])
    else:
        model = load_model(model_dict[0])
    file = request.files["file"]
    image_filename = file.filename
    file.save(os.path.join('static', image_filename))
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] \
    if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape \
    [1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(img)[0]
    labels = (pred > 0.5).astype(int)
    print(labels)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    return predict_result(chosen_model, runtimes, respon_model, image_filename)

def predict_result(model, run_time, probs, image_filename):
    class_list = {'Paper': 0, 'Rock': 1, 'Scissors': 2}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/result_select.html', labels=labels,
                            probs=probs, model=model, pred=idx_pred,
                            run_time=run_time, image_filename=image_filename)
    
if __name__ == "__main__":
        app.run(debug=True, host='0.0.0.0', port=2000)
