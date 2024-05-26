"""
This is the main app that loads Yolov5 for real time video inferencing. 
This app checks if an object has been detected by a detector object (fruits in our use case).
If a fruit is detected, the code proceeds to generate a speech using the OpenAI API. It creates a unique filename for the speech file by appending the current timestamp to the filename. The speech is generated with the text "Hmm, I see a fruit, do you?" using the OpenAI API's text-to-speech functionality.
The generated speech is then downloaded and saved to a file path specified by speech_file_path. The response.stream_to_file() method is used to download the file from the URL response and save it to the specified path.
Lastly, the code returns a JSON response using the jsonify() function. If an object is detected, it returns the URL of the generated speech file with a cache-busting parameter appended to it. The cache-busting parameter ensures that the latest version of the speech file is always fetched. If no object is detected, it returns an empty hint.
"""
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from openai import OpenAI
import time
from flask_socketio import SocketIO
import torch
from flask import Flask, render_template , Response, jsonify

app = Flask(__name__)
openai_api_key = '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'
client = OpenAI(api_key=openai_api_key)

app.config['STATIC_FOLDER'] = 'static/images'  # Set the static folder for storing images
static_folder = Path(app.config['STATIC_FOLDER'])
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO for real-time communication

class ObjectDetection:

    def __init__(self):
        # Load the YOLOv5 model
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path="./model_weights/yolov5.pt", force_reload=True)
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        self.detected_object_class = "No object detected"
        self.detected_object_description = "No description available."
        self.detected_object_images = {}
        self.object_detected = False


    def get_description(self, object_class):
                # Descriptions for various fruit classes
        descriptions = {
            "grapes": "Grapes are small, juicy fruits that grow in bunches. They can be red, green, or purple, and are sweet to eat. You can eat them fresh or use them to make juice, jelly, or raisins.",
            "apple": "Apples are crunchy and sweet fruits that come in many colors like red, green, and yellow. They are very healthy and make a great snack. You can eat them raw or use them to make apple pie or apple juice.",
            "banana": "Bananas are long, yellow fruits that are very sweet and soft inside. They are easy to peel and make a perfect snack. You can also use them in smoothies or to make banana bread.",
            "mango": "Mangoes are tropical fruits that are sweet and juicy. They have a large pit inside and come in different colors like yellow, orange, and red. Mangoes are delicious in smoothies, salads, or just by themselves.",
            "watermelon": "Watermelons are big, green fruits with a sweet, red inside. They are very juicy and perfect for hot days. You can eat watermelon slices or make refreshing watermelon juice.",
            "orange": "Oranges are round, orange fruits that are very juicy and sweet. They are full of vitamin C and are great for snacks or for making orange juice."
        }
        return descriptions.get(object_class, "Description not available.")

    def get_images(self, object_class):
        # File paths for images of different fruit classes

        images = {
            "grapes": {
                "whole": str(static_folder/ f"grapes.gif"),
                "sliced": str(static_folder/ f"grapes.webp")
            },
            "apple": {
                "whole": str(static_folder/ f"apple.gif"),
                "sliced": str(static_folder/ f"apple.jpeg")
            },
            "banana": {
                "whole": str(static_folder/ f"banana.gif"),
                "sliced": str(static_folder/ f"banana.png")
            },
            "mango": {
               "whole": str(static_folder/ f"mangoes.gif"),
                "sliced": str(static_folder/ f"mangoes.webp")
            },
            "watermelon": {
               "whole": str(static_folder/ f"watermelon.gif"),
                "sliced": str(static_folder/ f"watermelonsliced.webp")
            },
            "orange": {
              "whole": str(static_folder/ f"oranges.gif"),
                "sliced": str(static_folder/ f"oranges.jpeg")
            }
        }
        return images.get(object_class, {"whole": str(static_folder/ f"nd.png"), "sliced": str(static_folder/ f"nd.png")})

    def detect_objects(self, frame):
                # Convert the frame to an image and run the detection model
        img = Image.open(io.BytesIO(frame))
        results = self.model(img, size=640)
        # Check if any objects are detected

        if results.xyxy[0].shape[0] > 0:
            self.detected_object_class = results.names[int(results.xyxy[0][0][5])]
            self.detected_object_description = self.get_description(self.detected_object_class)
            self.detected_object_images = self.get_images(self.detected_object_class)
            self.object_detected = True
            socketio.emit('fruit_detected', {'detected': True})
            print(f"Detected object: {self.detected_object_class}")
        else:
            self.detected_object_class = "No object detected"
            self.detected_object_description = "No description available."
            self.detected_object_images = {"whole": str(static_folder/ f"nd.png"), "sliced": str(static_folder/ f"nd.png")}
            self.object_detected = False
            socketio.emit('fruit_detected', {'detected': False})
            print("No object detected")
           
            # Render the detection results on the image

        img = np.squeeze(results.render())
        
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        return frame

detector = ObjectDetection()

def gen():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    while cap.isOpened():
        success, frame = cap.read()  # Capture a frame from the webcam
        if success:
            ret, buffer = cv2.imencode('.jpg', frame)  # Encode the frame as JPEG
            frame = buffer.tobytes()  # Convert the encoded image to bytes
            frame = detector.detect_objects(frame)  # Detect objects in the frame
            # Stream the frame as a multipart HTTP response
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break  #

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/apppage')
def adventure():
    return render_template('apppage.html')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    # Ensure the static folder exists
    static_folder.mkdir(exist_ok=True)

    # Generate a unique filename based on the current timestamp
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

@app.route('/get_detected_object', methods=['GET'])
def get_detected_object():
    
    # Return the URL of the generated speech file with a cache-busting parameter
    # Append a query parameter with a unique value (e.g., timestamp)
    return jsonify({
        'description': f"{str(detector.detected_object_description)}"
    })


@app.route('/get_detected_fruits', methods=['GET'])
def get_detected_fruits():
    return jsonify(detector.detected_object_images)

@app.route('/hint', methods=['GET'])
def hint():
    if detector.object_detected:
        static_folder.mkdir(exist_ok=True)
        unique_filename = f"speech_{int(time.time())}.mp3"
        speech_file_path = static_folder / unique_filename
        # Generate speech using OpenAI API
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input="Hmm, I see a fruit, do you?",
            response_format="mp3"
        )

        # Download the file from the URL response
        response.stream_to_file(speech_file_path)
        
        # Return the URL of the generated speech file with a cache-busting parameter
        # Append a query parameter with a unique value (e.g., timestamp)
        return jsonify({
            'speech_url': f"{str(speech_file_path)}?v={int(time.time())}"
        })
    return jsonify({'hint': ''})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)






