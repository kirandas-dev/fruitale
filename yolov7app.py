import torch
import os
import io
from PIL import Image
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
from pathlib import Path
from openai import OpenAI
import time

app = Flask(__name__)
openai_api_key = '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'
client = OpenAI(api_key=openai_api_key)

app.config['STATIC_FOLDER'] = 'static/images'  # Set the static folder for storing images
static_folder = Path(app.config['STATIC_FOLDER'])
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO for real-time communication

class ObjectDetection:

    def __init__(self):
        # Load the YOLOv5 model
        try:
            self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', './model_weights/yolov7.pt', force_reload=True, trust_repo=True)
            self.model.eval()
            self.model.conf = 0.6  # confidence threshold (0-1)
            self.model.iou = 0.45  # NMS IoU threshold (0-1)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
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
            "grapes": {"whole": str(static_folder / "grapes.gif"), "sliced": str(static_folder / "grapes.webp")},
            "apple": {"whole": str(static_folder / "apple.gif"), "sliced": str(static_folder / "apple.jpeg")},
            "banana": {"whole": str(static_folder / "banana.gif"), "sliced": str(static_folder / "banana.png")},
            "mango": {"whole": str(static_folder / "mangoes.gif"), "sliced": str(static_folder / "mangoes.webp")},
            "watermelon": {"whole": str(static_folder / "watermelon.gif"), "sliced": str(static_folder / "watermelonsliced.webp")},
            "orange": {"whole": str(static_folder / "oranges.gif"), "sliced": str(static_folder / "oranges.jpeg")}
        }
        return images.get(object_class, {"whole": str(static_folder/ f"nd.png"), "sliced": str(static_folder/ f"nd.png")})
    def detect_objects(self, frame):
                # Convert the frame to an image and run the detection model

        try:
            # Open the image from bytes
            img = Image.open(io.BytesIO(frame))
            
            # Convert the image to a writable NumPy array (RGB format for the model)
            img_array = np.array(img)
            img_array.setflags(write=1)  # Ensure the NumPy array is writable

            # Perform object detection
            results = self.model(img_array, size=640)

            # Check if any objects are detected
            if results.xyxy[0].shape[0] > 0:
                self.detected_object_class = results.names[int(results.xyxy[0][0][5])]
                self.detected_object_description = self.get_description(self.detected_object_class)
                self.detected_object_images = self.get_images(self.detected_object_class)
                self.object_detected = True
                socketio.emit('fruit_detected', {'detected': True})
                print(f"Detected object: {self.detected_object_class}")
            else:
                self.reset_detection()
                socketio.emit('fruit_detected', {'detected': False})
                print("No object detected")

            # Render the detection results on the image
            img_with_boxes = np.squeeze(results.render())
            img_with_boxes.setflags(write=1)  # Ensure the NumPy array is writable

            # Convert the rendered image to BGR format for OpenCV
            img_BGR = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

            # Encode the frame back to bytes
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            return frame
        except Exception as e:
            print(f"Error detecting objects: {e}")
            self.reset_detection()
            return frame


    def reset_detection(self):
        self.detected_object_class = "No object detected"
        self.detected_object_description = "No description available."
        self.detected_object_images = {"whole": str(static_folder/ f"nd.png"), "sliced": str(static_folder/ f"nd.png")}
        self.object_detected = False

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

@app.route('/apppage')
def adventure():
    return render_template('apppage.html')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    static_folder.mkdir(exist_ok=True)
    unique_filename = f"speech_{int(time.time())}.mp3"
    speech_file_path = static_folder / unique_filename
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=detector.detected_object_description,
        response_format="mp3"
    )
    response.stream_to_file(speech_file_path)
    return jsonify({'speech_url': f"{str(speech_file_path)}?v={int(time.time())}"})

@app.route('/get_detected_object', methods=['GET'])
def get_detected_object():
    return jsonify({'description': f"{str(detector.detected_object_description)}"})

@app.route('/get_detected_fruits', methods=['GET'])
def get_detected_fruits():
    return jsonify(detector.detected_object_images)

@app.route('/hint', methods=['GET'])
def hint():
    if detector.object_detected:
        static_folder.mkdir(exist_ok=True)
        unique_filename = f"speech_{int(time.time())}.mp3"
        speech_file_path = static_folder / unique_filename
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input="Hmm, I see a fruit, do you?",
            response_format="mp3"
        )
        response.stream_to_file(speech_file_path)
        return jsonify({'speech_url': f"{str(speech_file_path)}?v={int(time.time())}"})
    return jsonify({'hint': ''})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
