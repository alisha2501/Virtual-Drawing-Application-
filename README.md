

# Virtual Drawing Application

This is a Python-based virtual drawing application that allows users to draw on a canvas using hand gestures. The project leverages OpenCV for image processing, Mediapipe for hand-tracking, and Numpy for array manipulation.

## Features
- **Hand Gesture Recognition:** Detects hand landmarks using Mediapipe.
- **Real-time Drawing:** Draw on the screen in real-time by moving your finger in front of the camera.
- **Color Selection:** Switch between different colors (Blue, Green, Red, Yellow) using specific hand gestures.
- **Clear Canvas:** Easily clear the canvas by making a gesture.
- **User Interface:** Basic UI elements such as buttons for color selection and canvas clearing.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe
- Numpy

You can install the required libraries using:
```bash
pip install opencv-python mediapipe numpy
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/alisha2501/virtual-drawing-app.git
   cd virtual-drawing-application
   ```

2. Run the script:
   ```bash
   python virtual_drawing.py
   ```
3. Use the webcam to detect hand gestures. Move your hand in front of the camera to start drawing on the screen. Use the top bar to select different colors or clear the canvas
