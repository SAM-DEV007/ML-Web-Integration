from pathlib import Path
from django.conf import settings

import mediapipe as mp
import numpy as np
import tensorflow as tf

import cv2
import copy
import itertools
import os
import time

import base64
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import StopConsumer


class predict():
    def __init__(self, model_path = str(Path(__file__).resolve().parent / 'HandGesture_Model/Model.tflite'), num_threads = 1):
        self.interpreter = tf.lite.Interpreter(model_path = model_path, num_threads = num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        
        if np.squeeze(result)[result_index] < 0.6: result_index = 3

        return result_index


mphands = mp.solutions.hands
hands = mphands.Hands()

model_predict = predict()


class HandGesture(AsyncWebsocketConsumer):
    async def connect(self):
        self.loop = asyncio.get_running_loop()
        await self.default()

        await self.accept()
    

    async def disconnect(self, close_code):
        await self.default()
        self.stop = True

        raise StopConsumer()


    async def default(self):
        self.show_palm = True
        self.show_ok = True
        self.show_peace = True

        self.img_busy = False

        self.model_db = 0
        self.gesture = None

        self.mphands = mphands
        self.hands = hands

        self.model_predict = model_predict

        self.mask_big_img = None
        self.big_img = None
        self.big_img_time = None

        self.mask_palm_img_holder = None
        self.mask_ok_img_holder = None
        self.mask_peace_img_holder = None

        self.palm_img_holder = None
        self.ok_img_holder = None
        self.peace_img_holder = None

        self.temp_mask_palm_img_holder = None
        self.temp_mask_ok_img_holder = None
        self.temp_mask_peace_img_holder = None

        self.temp_palm_img_holder = None
        self.temp_ok_img_holder = None
        self.temp_peace_img_holder = None

        self.img_size = 150

        self.palm_path = str(settings.BASE_DIR / 'instagram_filters/HandGesture_Data/Palm.png')
        self.ok_path = str(settings.BASE_DIR / 'instagram_filters/HandGesture_Data/Ok.png')
        self.peace_path = str(settings.BASE_DIR / 'instagram_filters/HandGesture_Data/Peace.png')

        self.size = 100

        self.palm = await self.loop.run_in_executor(None, cv2.imread, self.palm_path)
        self.palm = await self.loop.run_in_executor(None, cv2.resize, self.palm, (self.size, self.size))
        self.mask_palm = await self.mask(self.palm)
        
        self.ok = await self.loop.run_in_executor(None, cv2.imread, self.ok_path)
        self.ok = await self.loop.run_in_executor(None, cv2.resize, self.ok, (self.size, self.size))
        self.mask_ok = await self.mask(self.ok)

        self.peace = await self.loop.run_in_executor(None, cv2.imread, self.peace_path)
        self.peace = await self.loop.run_in_executor(None, cv2.resize, self.peace, (self.size, self.size))
        self.mask_peace = await self.mask(self.peace)


    async def landmark_list(self, image, landmarks):
        '''Generates the landmark list'''

        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return await self.pre_process_data(landmark_point)


    async def pre_process_data(self, landmark_list):
        '''Pre processes the data for the trained model'''

        temp_landmark_list = await self.loop.run_in_executor(None, copy.deepcopy, landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list


    async def detect_hands(self, frame, hands, model_predict):
        '''Detects the hand and returns the suitable gesture'''

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_landmarks = result.multi_hand_landmarks
        gesture = None

        if hand_landmarks:
            handLMs = hand_landmarks[0]
            if handLMs:
                landmark = await self.landmark_list(frame, handLMs)
                gesture = model_predict(landmark)
        
        if gesture == 3: gesture = None
        return gesture


    async def mask(self, img):
        '''Masks the image'''

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

        return mask


    async def pre_img(self, frame, size, img, mask_img, gesture):
        '''Initializes the hand img to the frame'''

        img_section = [-600, -360, -130]
        img_x = img_section[gesture]

        roi = frame[-465: -465+size, img_x: img_x+size]
    
        roi[np.where(mask_img)] = 0
        roi += img


    async def get_img(self, frame):
        '''Gets the images after the gesture'''

        cv2.rectangle(frame, [130, 170], [510, 450], (0, 0, 0), 2)

        cv2.rectangle(frame, [110, 150], [530, 170], (255, 255, 255), -1)
        cv2.rectangle(frame, [110, 450], [530, 470], (255, 255, 255), -1)
        cv2.rectangle(frame, [110, 170], [130, 450], (255, 255, 255), -1)
        cv2.rectangle(frame, [510, 170], [530, 450], (255, 255, 255), -1)

        cv2.rectangle(frame, [110, 150], [530, 470], (0, 0, 0), 2)
            
        img = frame[149: 472, 109: 532] 

        return img


    async def hold_img_main(self, frame, img, mask_img):
        '''Holds the img on the screen for a while'''

        roi = frame[149: 472, 109: 532]
        roi[np.where(mask_img)] = 0

        roi += img


    async def change_img(self, frame, img, mask_img, gesture: int, img_size: int):
        '''Changes the gesture image'''

        img_section = [-600, -400, -200]
        img_x = img_section[gesture]

        roi = frame[-465: -465+img_size, img_x: img_x+img_size]
        roi[np.where(mask_img)] = 0

        roi += img


    async def receive(self, bytes_data):
        '''The main function to start the filter'''

        if bytes_data:
            self.frame = await self.loop.run_in_executor(None, cv2.imdecode, np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.frame = await self.loop.run_in_executor(None, cv2.flip, self.frame, 1)

            if self.frame.shape[0] != 480 or self.frame.shape[1] != 640: 
                self.frame = await self.loop.run_in_executor(None, cv2.resize, self.frame, (640, 480))

            if self.show_palm: await self.pre_img(self.frame, self.size, self.palm, self.mask_palm, 0)
            else: await self.change_img(self.frame, self.palm_img_holder, self.mask_palm_img_holder, 0, self.img_size)
            if self.show_ok: await self.pre_img(self.frame, self.size, self.ok, self.mask_ok, 1)
            else: await self.change_img(self.frame, self.ok_img_holder, self.mask_ok_img_holder, 1, self.img_size)
            if self.show_peace: await self.pre_img(self.frame, self.size, self.peace, self.mask_peace, 2)
            else: await self.change_img(self.frame, self.peace_img_holder, self.mask_peace_img_holder, 2, self.img_size)

            if self.big_img is not None:
                if (time.time() - self.big_img_time) > 1.5:
                    
                    self.img_busy = False
                    self.model_db = 0
                    self.big_img_time = None
                    self.big_img = None
                    self.mask_big_img = None
                else:
                    await self.hold_img_main(self.frame, self.big_img, self.mask_big_img) 
            
            if self.big_img is None:
                if self.temp_palm_img_holder is not None:
                    self.palm_img_holder = self.temp_palm_img_holder
                    self.mask_palm_img_holder = self.temp_mask_palm_img_holder

                    self.temp_mask_palm_img_holder = None
                    self.temp_palm_img_holder = None
                    self.show_palm = False

                if self.temp_ok_img_holder is not None:
                    self.ok_img_holder = self.temp_ok_img_holder
                    self.mask_ok_img_holder = self.temp_mask_ok_img_holder
                    
                    self.temp_mask_ok_img_holder = None
                    self.temp_ok_img_holder = None
                    self.show_ok = False

                if self.temp_peace_img_holder is not None:
                    self.peace_img_holder = self.temp_peace_img_holder
                    self.mask_peace_img_holder = self.temp_mask_peace_img_holder

                    self.temp_mask_peace_img_holder = None
                    self.temp_peace_img_holder = None
                    self.show_peace = False
            
            if not self.model_db: self.model_db = time.time()
            
            if not self.img_busy:
                self.gesture = await self.detect_hands(self.frame[165: 480, 0: 640], self.hands, self.model_predict)

            if self.gesture is not None and (time.time() - self.model_db) > 1 and not self.img_busy:
                self.img_busy = True 
                self.temp_frame = await self.loop.run_in_executor(None, copy.deepcopy, self.frame)

                self.img = await self.get_img(self.temp_frame)
                self.img = cv2.detailEnhance(self.img, sigma_r=0.15, sigma_s=3) 
                
                self.m = await self.mask(self.img)
                await self.hold_img_main(self.frame, self.img, self.m)

                self.big_img_time = time.time()
                self.big_img = self.img
                self.mask_big_img = self.m
                
                self.img = await self.loop.run_in_executor(None, cv2.resize, self.img, (self.img_size, self.img_size))
                if self.gesture == 0: 
                    self.temp_palm_img_holder = self.img
                    self.temp_mask_palm_img_holder = await self.mask(self.img)
                elif self.gesture == 1: 
                    self.temp_ok_img_holder = self.img
                    self.temp_mask_ok_img_holder = await self.mask(self.img)
                elif self.gesture == 2:
                    self.temp_peace_img_holder = self.img
                    self.temp_mask_peace_img_holder = await self.mask(self.img)
                self.gesture = None

            self.buffer_img = await self.loop.run_in_executor(None, cv2.imencode, '.jpg', self.frame)
            self.b64_img = base64.b64encode(self.buffer_img[1]).decode('utf-8')

            await self.send(self.b64_img)
        else:
            await self.default()
            await self.close()