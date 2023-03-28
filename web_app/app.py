import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL.Image as Image


from flask import Flask, request, render_template

app = Flask(__name__)

# Load your Keras model here
path = r'../models/test_generator.h5'
model = keras.models.load_model(path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_image', methods=['GET'])
def generate_image():

    noise = np.random.normal(0, 1, (1, 100))
    gen_imgs = model.predict(noise)

    gen_imgs = 0.5 * gen_imgs + 0.5
    fig = gen_imgs[0]
    img = Image.fromarray((fig * 255).astype(np.uint8))

    # Convert the image to a JSON response
    response = {
        'image': f'data:image/png;base64,{to_base64(img)}'
    }
    return json.dumps(response)

def to_base64(img):
    """Convert a PIL image to a base64 string."""
    import base64
    from io import BytesIO

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

if __name__ == '__main__':
    app.run()
