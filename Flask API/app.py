import io
import json
from urllib.request import urlopen, Request

import requests
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras

app = Flask(__name__)
CORS(app)


def convert(url):
    request_site = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    image_url = urlopen(request_site)
    image_bytes = io.BytesIO(image_url.read())
    image = Image.open(image_bytes)
    return image


@app.route("/", methods=["POST"])
def main():
    json_data = request.get_json()
    image = convert(json_data['url']).resize((300, 300))
    # testing local image
    # image = keras.preprocessing.image.image_utils.load_img("Test/IMG_2872.JPEG", target_size = (256, 256))
    image = keras.preprocessing.image.img_to_array(image)
    request_message = json.dumps({"model": "ericchi_7728de", "input": image.tolist(), "verbose": 1})
    inference_url = "https://flashai.io/serve"
    request_headers = {"content-type": "application/json"}
    response = requests.post(inference_url, headers=request_headers, data=request_message)
    prediction = response.content.decode("ascii", "ignore")
    probability = prediction.replace("[", "").replace("]", "").split()
    return jsonify(probability)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
