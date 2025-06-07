from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from detect_image import detect_mask

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img_bytes = file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    results = detect_mask(img)

    # Save and return image
    output_path = 'result.jpg'
    cv2.imwrite(output_path, results['visualization'])
    return send_file(output_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)