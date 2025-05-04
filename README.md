# Hand Tracking with MediaPipe and OpenCV 🖐️

This project demonstrates real-time hand tracking using [MediaPipe](https://google.github.io/mediapipe/) and [OpenCV](https://opencv.org/) in Python. It provides a simple and extensible class, `HandDetector`, to detect hands, get landmark positions, determine which fingers are up, and calculate distances between landmarks.

---

## 🔧 Features

* Real-time hand tracking using a webcam
* Get precise hand landmark positions
* Determine which fingers are up
* Calculate Euclidean distance between two hand landmarks
* Draw hand skeletons and bounding boxes on the frame

---

## 📸 Dependencies

* Python 3.7+
* OpenCV
* MediaPipe

Install the dependencies with:

```bash
pip install opencv-python mediapipe
```

---

## 🧠 How It Works

The `HandDetector` class wraps MediaPipe's `Hands` module with easy-to-use methods:

* `findHands(img)`: Detects and optionally draws hands on the input image.
* `findPosition(img)`: Returns list of landmark positions and bounding box.
* `fingersUp()`: Returns a list indicating which fingers are raised.
* `findDistance(p1, p2, img)`: Computes distance between two landmarks.

---


---

## 📁 Project Structure

```
.
├── hand_tracking_module.py   # Contains the HandDetector class
├── requirements.txt          # Contains necessary packages
└── README.md                 # You are here
```

---

