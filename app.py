"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from openai import OpenAI

import torch
from flask import Flask, render_template, request, redirect, Response, jsonify

app = Flask(__name__)
openai_api_key = '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'
client = OpenAI(api_key=openai_api_key)

class ObjectDetection:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path="./best.pt", force_reload=True)
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        self.detected_object_class = "No object detected"
        self.detected_object_description = "No description available."

    def get_description(self, object_class):
        descriptions = {
            "grapes": "Grapes are small, juicy fruits that grow in bunches. They can be red, green, or purple, and are sweet to eat. You can eat them fresh or use them to make juice, jelly, or raisins.",
            "apple": "Apples are crunchy and sweet fruits that come in many colors like red, green, and yellow. They are very healthy and make a great snack. You can eat them raw or use them to make apple pie or apple juice.",
            "banana": "Bananas are long, yellow fruits that are very sweet and soft inside. They are easy to peel and make a perfect snack. You can also use them in smoothies or to make banana bread.",
            "mango": "Mangoes are tropical fruits that are sweet and juicy. They have a large pit inside and come in different colors like yellow, orange, and red. Mangoes are delicious in smoothies, salads, or just by themselves.",
            "watermelon": "Watermelons are big, green fruits with a sweet, red inside. They are very juicy and perfect for hot days. You can eat watermelon slices or make refreshing watermelon juice.",
            "orange": "Oranges are round, orange fruits that are very juicy and sweet. They are full of vitamin C and are great for snacks or for making orange juice."
        }
        return descriptions.get(object_class, "Description not available.")


    def detect_objects(self, frame):
        img = Image.open(io.BytesIO(frame))
        results = self.model(img, size=640)

        if results.xyxy[0].shape[0] > 0:
            self.detected_object_class = results.names[int(results.xyxy[0][0][5])]
            self.detected_object_description = self.get_description(self.detected_object_class)
            print(f"Detected object: {self.detected_object_class}")
        else:
            self.detected_object_class = "No object detected"
            self.detected_object_description = "No description available."
            print("No object detected")

        img = np.squeeze(results.render())
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        return frame

detector = ObjectDetection()

def gen():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            frame = detector.detect_objects(frame)

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/index')
def adventure():
    return render_template('index.html')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.config['STATIC_FOLDER'] = 'static'
static_folder = Path(app.config['STATIC_FOLDER'])

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    # Ensure the static folder exists
    static_folder.mkdir(exist_ok=True)

    # Generate a unique filename based on the current timestamp
    import time
    unique_filename = f"speech_{int(time.time())}.mp3"
    speech_file_path = static_folder / unique_filename
    print("Description of the fruit detected", detector.detected_object_description)

    # Generate speech using OpenAI API
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=detector.detected_object_description,
        response_format="mp3"
    )

    # Download the file from the URL response
    response.stream_to_file(speech_file_path)
    
    # Return the URL of the generated speech file with a cache-busting parameter
    # Append a query parameter with a unique value (e.g., timestamp)
    return jsonify({
        'speech_url': f"{str(speech_file_path)}?v={int(time.time())}"
    })



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
