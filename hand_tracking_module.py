import cv2
import mediapipe as mp
import math

# Tip landmark IDs for each finger (thumb to pinky)
tipIds = [4, 8, 12, 16, 20]

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initialize the Hand Detector using MediaPipe.
        :param mode: Whether to treat input images as static.
        :param maxHands: Maximum number of hands to detect.
        :param detectionCon: Minimum detection confidence.
        :param trackCon: Minimum tracking confidence.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.lmList = []

    def findHands(self, img, draw=True):
        """
        Detect hands and draw landmarks on the image.
        :param img: Image in BGR format.
        :param draw: Whether to draw the landmarks.
        :return: Image with drawn landmarks (if draw=True).
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Get the list of landmark positions for a specific hand.
        :param img: Image.
        :param handNo: Index of the hand to use (0 or 1).
        :param draw: Whether to draw the landmark positions.
        :return: List of landmarks, bounding box.
        """
        xList, yList = [], []
        bbox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = [xmin, ymin, xmax, ymax]
                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self):
        """
        Determine which fingers are up based on landmark positions.
        :return: List with binary values (1 for up, 0 for down) for each finger.
        """
        fingers = []
        if not self.lmList:
            return fingers

        # Thumb (check horizontal direction)
        if self.lmList[tipIds[0]][1] > self.lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other four fingers (check vertical direction)
        for id in range(1, 5):
            if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        """
        Calculate the Euclidean distance between two landmarks.
        :param p1: Landmark index 1.
        :param p2: Landmark index 2.
        :param img: Image to draw on.
        :param draw: Whether to draw the distance line.
        :return: Tuple (distance, image, coordinates).
        """
        if not self.lmList:
            return 0, img, []

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
