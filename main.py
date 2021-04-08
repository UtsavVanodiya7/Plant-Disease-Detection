import os
import traceback

import cv2
import numpy as np

from keras.models import load_model
from flask import flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from app import app

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}
model = load_model('model' + os.sep + 'inception_v3')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


def predict(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
    img = np.reshape(resized_img, (1, 128, 128, 3))
    prediction = model.predict(img)
    max_index = np.argmax(prediction)
    print(prediction)

    if max_index == 1:
        return "No Disease"
    else:
        return "Has Disease"


@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['file']

    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            prediction = predict(image_path)
            return render_template('index.html', filename=filename, prediction=prediction)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    except:
        traceback.print_exc()


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()
