import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import set_session

from flask import (Flask, current_app, flash, redirect, render_template,
                   request, url_for, send_from_directory)
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join(os.path.join(os.getcwd(), "data"), 'uploads')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

global sess
global graph
sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
model = tf.keras.models.load_model('kerasmodel.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

validation_image_generator = ImageDataGenerator(rescale=1./255)

IMG_HEIGHT = 150
IMG_WIDTH = 150

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def testImage(yourImg_dir):
    with graph.as_default():
      set_session(sess)
      data = validation_image_generator.flow_from_directory(batch_size=2,
        directory=yourImg_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary')
      prediction = model.predict(data)
      return prediction
    return 0.14386

@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      return redirect(url_for('uploaded_file',
        filename=filename))
  return '''
  <!doctype html>
  <title>Save the Bees!</title>
  <h1>Is your bee healthy?</h1>
  <form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <input type=submit value=Upload>
  </form>
  '''
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    #return send_from_directory(app.config['UPLOAD_FOLDER'],
                               #filename)
    with graph.as_default():
      return str(testImage(app.config['UPLOAD_FOLDER'])[0][0])
    return '''
    <!doctype html>
    <h1>An error has occured</h1>
    '''
