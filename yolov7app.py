import torch
import os
import io
from PIL import Image
import numpy as np
import cv2
from flask import Flask, render_template, request, Response, jsonify
from flask_socketio import SocketIO
from pathlib import Path
from openai import OpenAI
import time

app = Flask(__name__)
openai_api_key = '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'
client = OpenAI(api_key=openai_api_key)
app.config['STATIC_FOLDER'] = 'static'
static_folder = Path(app.config['STATIC_FOLDER'])
socketio = SocketIO(app, cors_allowed_origins="*")

class ObjectDetection:

    def __init__(self):
        try:
            #self.model = torch.hub.load("WongKinYiu/yolov7", "custom", path="./best.pt", force_reload=True)
            self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', './model_weights/yolov7.pt',force_reload=True, trust_repo=True)
            #self.model = custom(path_or_model='yolov7.pt')  # custom example
# model = create(name='yolov7', pretrained=True, channels=3, classes=80, autoshape=True)  # pretrained example

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
        descriptions = {
            "grapes": "Grapes are small, juicy fruits that grow in bunches...",
            "apple": "Apples are crunchy and sweet fruits that come in many colors...",
            "banana": "Bananas are long, yellow fruits that are very sweet...",
            "mango": "Mangoes are tropical fruits that are sweet and juicy...",
            "watermelon": "Watermelons are big, green fruits with a sweet...",
            "orange": "Oranges are round, orange fruits that are very juicy..."
        }
        return descriptions.get(object_class, "Description not available.")

    def get_images(self, object_class):
        images = {
            "grapes": {"whole": str(static_folder / "grapes.gif"), "sliced": str(static_folder / "grapes.webp")},
            "apple": {"whole": str(static_folder / "apple.gif"), "sliced": str(static_folder / "apple.jpeg")},
            "banana": {"whole": str(static_folder / "banana.gif"), "sliced": str(static_folder / "banana.png")},
            "mango": {"whole": str(static_folder / "mangoes.gif"), "sliced": str(static_folder / "mangoes.webp")},
            "watermelon": {"whole": str(static_folder / "watermelon.gif"), "sliced": str(static_folder / "watermelonsliced.webp")},
            "orange": {"whole": str(static_folder / "oranges.gif"), "sliced": str(static_folder / "oranges.jpeg")}
        }
        return images.get(object_class, {"whole": "https://example.com/default_whole.jpg", "sliced": "https://example.com/default_sliced.jpg"})

    def detect_objects(self, frame):
        try:
            img = Image.open(io.BytesIO(frame))
            results = self.model(img, size=640)

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

            img = np.squeeze(results.render())            
            img.setflags(write=1)  # Ensure the NumPy array is writable

            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            return frame
        except Exception as e:
            print(f"Error detecting objects: {e}")
            self.reset_detection()
            return frame

    def reset_detection(self):
        self.detected_object_class = "No object detected"
        self.detected_object_description = "No description available."
        self.detected_object_images = {"whole": "https://example.com/default_whole.jpg", "sliced": "https://example.com/default_sliced.jpg"}
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

@app.route('/index')
def adventure():
    return render_template('index.html')

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