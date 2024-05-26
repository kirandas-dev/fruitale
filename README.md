# Wonderland of Fruits

Welcome to the Wonderland of Fruits! This interactive web application detects fruit objects through a video feed and provides hints through audio messages when fruits are detected continuously for a specific period.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

The Wonderland of Fruits application uses a video feed to detect fruit objects in real-time. When a fruit is detected continuously for 5 seconds, the app fetches a hint from the server and plays an audio message. The app also provides options to display and interact with images of the detected fruits.

## Features

- Real-time fruit detection using a video feed
- Audio hints triggered by continuous fruit detection
- Interactive display of detected fruit images
- Toggle between whole and sliced fruit images

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript, Socket.IO, jQuery
- **Backend**: Python, Flask
- **API**: OpenAI API for generating speech
- **Other**: Socket.IO for real-time communication, Flask for serving static files and handling requests

## Setup and Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/wonderland-of-fruits.git
    cd wonderland-of-fruits
    ```

2. **Install Dependencies**:
    Ensure you have Python and pip installed. Then, install the required Python packages:
    ```sh
    python3 -m venv .venv     
    source .venv/bin/activate  
    pip install -r requirements.txt
    ```

3. **Run the Flask App**:
    ```sh
    python yolov5app.py  # For YOLOv5
    python yolov7app.py  # For YOLOv7- Please run the script twice. Script fails to compile the first time. 
    ```

    Quick Note: 
    ```
    The app loads YOLOv5/7 for real-time video inferencing. 
    This app checks if an object has been detected by a detector object (fruits in our use case).
    If a fruit is detected, the code proceeds to generate a speech using the OpenAI API. It creates a unique filename for the speech file by appending the current timestamp to the filename. The speech is generated with the text "Hmm, I see a fruit, do you?" using the OpenAI API's text-to-speech functionality.
    The generated speech is then downloaded and saved to a file path specified by speech_file_path. The response.stream_to_file() method is used to download the file from the URL response and save it to the specified path.
    Lastly, the code returns a JSON response using the jsonify() function. If an object is detected, it returns the URL of the generated speech file with a cache-busting parameter appended to it. The cache-busting parameter ensures that the latest version of the speech file is always fetched. If no object is detected, it returns an empty hint.
    ```

4. **Access the Application**:
    Open your web browser and go to `http://127.0.0.1:5000`, this is our Homepage. 
    Open your web browser and go to `http://127.0.0.1:5000/apppage`, this is our App page. 

## Usage

1. **Real-Time Detection**:
   - The app will start detecting fruits through the video feed.
   - When a fruit is detected continuously for 5 seconds, an audio hint will be played.

2. **Interacting with Detected Fruits**:
   - Click the "Discover" button to see images of detected fruits.
   - Click the knife button on the fruit images to toggle between whole and sliced images.

3. **Viewing Object Descriptions**:
   - Click the "i" button to view a description of the detected object.

## Project Structure

```plaintext
wonderland-of-fruits/
├── yolov5app.py                # Main application script for YOLOv5
├── templates/
│   ├── apppage.html            # Main app page
│   ├── home.html               # Main home page
├── static/
│   ├── css/
│   │   ├── styles.css          # App page CSS
│   │   ├── homestyle.css       # Homepage CSS
│   ├── images/
│   │   ├── apple.jpg           # Repository to fetch fruit images
│   │   ├── banana.jpg
│   │   └── ...
├── model_weights/
│   ├── yolov5/                 # YOLOv5 model files and weights
│   ├── yolov7/                 # YOLOv7 model files and weights
├── .github/workflows/
│   └── main_fruital.yaml       # GitHub action script for Azure deployment
├── utils/
│   └── helpers.py              # Utility functions
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation