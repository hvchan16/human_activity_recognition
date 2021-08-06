# Importing Required Libraries
import numpy as np
import cv2
import tensorflow as tf
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from req_functions import mediapipe_detection, draw_styled_landmarks, mp_holistic, extract_key_points, mp_drawing, draw_landmarks

### Real Testing ###


class VideoCamera(object):

    # Function to capture the Video Frames through the Webcam

    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def __del__(self):
        self.cap.release()

    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        return output_frame

    def gen_frame(self):

        # New Detection Variables
        sequence = []
        sentence = []
        threshold = 0.7

        # Actions array
        actions = np.array(['hello', 'thanks', 'iloveyou'])
        # Colors array
        colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

        # Load Model
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions.shape[0], activation='softmax'))

        model.load_weights(r'action.h5')

        # Set Mediapipe Model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while self.cap.isOpened():
                # Read Feed
                ret, frame = self.cap.read()

                # Make Detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                # draw_landmarks
                draw_styled_landmarks(image, results)

                # Prediction Logic
                keypoints = extract_key_points(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])

                # Visualization Logic
                if (res[np.argmax(res)] > threshold).all:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # viz  probablities
                image = VideoCamera.prob_viz(res, actions, image, colors)

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Show to Screen
                cv2.imshow('Action Prediction (Press "Q" to Exit!)', image)
                # Brake Gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            self.cap.release()

    def clseWeb(self):
        self.cap.release()

if __name__ == "__main__":
    VideoCamera().gen_frame()
