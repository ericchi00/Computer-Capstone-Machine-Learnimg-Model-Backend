import io
import json
from urllib.request import urlopen, Request

from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from numpy import argmax
from tensorflow import keras, expand_dims, nn

app = Flask(__name__)
CORS(app)
model_dir = "./Model"
model = keras.models.load_model(model_dir, compile=False)


def convert(url):
    request_site = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    image_url = urlopen(request_site)
    image_bytes = io.BytesIO(image_url.read())
    img = Image.open(image_bytes).convert('RGB').resize((300, 300))
    return img


@app.route("/", methods=["POST"])
def main():
    json_data = request.get_json()
    img = convert(json_data['url'])
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = expand_dims(img_array, 0)
    prediction = model.predict(img_array).tolist()
    score = nn.softmax(prediction[0])
    class_probabilities = [argmax(score), prediction[0]]
    return json.dumps(class_probabilities, default=str)


if __name__ == "__main__":
    app.run(port=5000)
