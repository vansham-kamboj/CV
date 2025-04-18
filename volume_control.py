import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Pycaw setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

def generate_frames():
    cap = cv2.VideoCapture("http://192.168.31.196:4747/video")

    while True:
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((cx, cy))

                if len(lmList) >= 9:
                    x1, y1 = lmList[4]
                    x2, y2 = lmList[8]
                    length = math.hypot(x2 - x1, y2 - y1)

                    # Reduced sensitivity by expanding min range and smoothing
                    vol = np.interp(length, [50, 200], [volMin, volMax])
                    volume.SetMasterVolumeLevel(vol, None)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
