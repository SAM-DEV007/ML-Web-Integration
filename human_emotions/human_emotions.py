import tensorflow as tf
import mediapipe as mp
import numpy as np

import os
import cv2
import keras

import base64
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import StopConsumer

from pathlib import Path


face_detection = mp.solutions.face_detection.FaceDetection()
model_path = str(Path(__file__).resolve().parent / 'Model/HumanEmotions_Model.h5')

if not os.path.exists(model_path):
    from human_emotions import download_model
# model = keras.models.load_model(model_path)


class HumanEmotions(AsyncWebsocketConsumer):
    async def connect(self):
        self.loop = asyncio.get_running_loop()
        await self.default()

        await self.accept()
    

    async def disconnect(self, close_code):
        await self.default()
        self.stop = True

        raise StopConsumer()
    

    async def default(self):
        self.model = keras.models.load_model(model_path) # May slow down the process, but to ensure that the website doesn't shut down
        self.h, self.w = 480, 640
        self.emo = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']


    async def receive(self, bytes_data):
        if bytes_data:
            frame = await self.loop.run_in_executor(None, cv2.imdecode, np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            frame = await self.loop.run_in_executor(None, cv2.flip, frame, 1)

            face_frame = await self.loop.run_in_executor(None, cv2.cvtColor, frame, cv2.COLOR_BGR2RGB)
            face = await self.loop.run_in_executor(None, face_detection.process, face_frame)

            if face.detections:
                for detection in face.detections:
                    face_data = detection.location_data
                    box_data = face_data.relative_bounding_box
                    x = round(box_data.xmin * self.w)
                    y = round(box_data.ymin * self.h)
                    wi = round(box_data.width * self.w)
                    he = round(box_data.height * self.h)
                
                    if 0 < x < self.w and 0 < y < self.h and 0 < x+wi < self.w and 0 < y+he < self.h:
                        pred_frame = frame[y: y+he, x: x+wi]
                        pred_frame = cv2.resize(pred_frame, (48, 48), interpolation=cv2.INTER_LINEAR_EXACT)

                        img = np.asarray(pred_frame)
                        img = np.expand_dims(img, axis=0)

                        prediction = np.round(np.squeeze(await self.loop.run_in_executor(None, self.model.predict, img)), 4)
                        predict = np.argmax(prediction)

                        cv2.rectangle(frame, (x, y), (x+wi, y+he), (0, 255, 0), 2)
                        cv2.putText(frame, self.emo[predict], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA, False)

            self.buffer_img = await self.loop.run_in_executor(None, cv2.imencode, '.jpg', frame)
            self.b64_img = base64.b64encode(self.buffer_img[1]).decode('utf-8')

            await self.send(self.b64_img)
        else:
            await self.default()
            await self.close()