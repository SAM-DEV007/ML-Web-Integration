import tensorflow as tf
import mediapipe as mp
import numpy as np

import os
import cv2
import keras

from django.conf import settings


if __name__ == '__main__':
    model_path = str(settings.BASE_DIR / 'human_emotions/Model/HumanEmotions_Model.h5')
    model = keras.models.load_model(model_path)

    h, w = 480, 640

    face_detection = mp.solutions.face_detection.FaceDetection()

    emo = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

    #
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)

    face_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = face_detection.process(face_frame)

    if face.detections:
        for detection in face.detections:
            face_data = detection.location_data
            box_data = face_data.relative_bounding_box
            x = round(box_data.xmin * w)
            y = round(box_data.ymin * h)
            wi = round(box_data.width * w)
            he = round(box_data.height * h)
        
            if 0 < x < w and 0 < y < h and 0 < x+wi < w and 0 < y+he < h:
                gray_frame = frame[y: y+he, x: x+wi]
                gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.resize(gray_frame, (48, 48))

                prediction = np.round(np.squeeze(model.predict(gray_frame)), 4)
                predict = np.argmax(prediction)

                cv2.rectangle(frame, (x, y), (x+wi, y+he), (0, 255, 0), 2)
                cv2.putText(frame, emo[predict], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA, False)