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
    pip install flask
    pip install openai
    pip install pathlib
    ```

3. **Run the Flask App**:
    ```sh
    python app.py
    ```

4. **Access the Application**:
    Open your web browser and go to `http://127.0.0.1:5000`.

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