import cv2
import os
import copy
import math
import random
import time

import mediapipe as mp
import numpy as np

import base64
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import StopConsumer

from django.conf import settings


mp_face_detection = mp.solutions.face_detection
face_detection = mp.solutions.face_detection.FaceDetection()

class FlappyBird(AsyncWebsocketConsumer):
    async def connect(self):
        self.loop = asyncio.get_running_loop()
        await self.default()

        await self.accept()
    

    async def disconnect(self, close_code):
        await self.default()
        self.stop = True

        raise StopConsumer()


    async def default(self):
        self.h, self.w = 480, 640

        self.mp_face_detection = mp_face_detection
        self.face_detection = face_detection

        self.nose_tip_x, self.nose_tip_y = None, None
        self.prev_nose_tip_x, self.prev_nose_tip_y = None, None
        self.prev_flappy = None

        self.fl = True

        self.pillar_val = 1
        self.pillar_num = 3

        self.pillar_list = [None]*self.pillar_num
        self.mask_pillar_list = [None]*self.pillar_num
        self.pillar_point_list = [False]*self.pillar_num

        self.pv_list = [525]*self.pillar_num
        self.pillar_resize = [0]*self.pillar_num

        self.flappy_coords = []

        self.flappy_size = 100
        self.angle = 0
        self.dest_angle = 0

        self.score = 0
        
        self.start = False
        self.finish = False

        self.buffer = time.time()

        self.flappy_path =  str(settings.BASE_DIR / 'instagram_filters/FlappyBird_Data/Flappy Bird.png')
        self.pillar_path =  str(settings.BASE_DIR / 'instagram_filters/FlappyBird_Data/Pillar.png')

        self.flappy = await self.loop.run_in_executor(None, cv2.imread, self.flappy_path)
        self.flappy = cv2.resize(self.flappy, (self.flappy_size, self.flappy_size))
        self.mask_flappy = await self.mask(self.flappy)

        self.originial_flappy = await self.loop.run_in_executor(None, copy.deepcopy, self.flappy)

        self.original_pillar = await self.loop.run_in_executor(None, cv2.imread, self.pillar_path)
        self.p_h, self.p_w, _ = self.original_pillar.shape


    async def mask(self, img):
        '''Masks the image'''

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

        return mask


    async def rotate_image(self, image, angle):
        '''Rotates the image by a certain angle'''

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = await self.loop.run_in_executor(None, cv2.getRotationMatrix2D, image_center, angle, 1.0)
        result = await self.loop.run_in_executor(None, cv2.warpAffine, image, rot_mat, image.shape[1::-1], cv2.INTER_LINEAR)

        return result


    async def calculate_angle(self, a, b, c) -> int:
        '''Calculates the angle'''

        angle = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        
        return round(angle)


    async def receive(self, bytes_data):
        if bytes_data:
            if (time.time() - self.buffer) > 3: 
                self.start = True

            self.frame = await self.loop.run_in_executor(None, cv2.imdecode, np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.frame = await self.loop.run_in_executor(None, cv2.flip, self.frame, 1)

            if self.frame.shape[0] != 480 or self.frame.shape[1] != 640: 
                self.frame = await self.loop.run_in_executor(None, cv2.resize, self.frame, (640, 480))

            self.face_frame = await self.loop.run_in_executor(None, copy.deepcopy, self.frame)

            self.face_frame = cv2.cvtColor(self.face_frame, cv2.COLOR_BGR2RGB)
            self.face = await self.loop.run_in_executor(None, self.face_detection.process, self.face_frame)

            if self.pv_list[self.pillar_val-1] <= 350 and self.fl:
                self.pillar_val += 1
            if self.pillar_val > len(self.pillar_list): 
                self.pillar_val = len(self.pillar_list)
                self.fl = False

            for i, pillar in enumerate(self.pillar_list[:self.pillar_val]):
                if pillar is None:
                    self.random_resize = random.randint(50, 260)
                    self.pillar_resize[i] = self.random_resize
                    pillar = await self.loop.run_in_executor(None, copy.deepcopy, self.original_pillar)

                    pillar = pillar[self.random_resize: self.random_resize+480, 0: self.p_w]
                    pillar = await self.loop.run_in_executor(None, cv2.resize, pillar, (self.p_w, self.h))
                    self.mask_pillar = await self.mask(pillar)
                
                self.pv_val = self.pv_list[i]

                self.pillar_exc_y = set(range((280-self.pillar_resize[i]), (490-self.pillar_resize[i])+1))
                self.pillar_all_y = set(range(0, self.h+1))

                self.pillar_x = list(range(self.pv_val+10, self.pv_val + 100))
                self.pillar_y = list(self.pillar_all_y - self.pillar_exc_y)

                if self.flappy_coords:
                    self.begin, self.end = self.flappy_coords
                    self.begin_x, self.begin_y = self.begin
                    self.end_x, self.end_y = self.end

                    self.flappy_x = list(range(self.begin_x+15, self.end_x-15))
                    self.flappy_y = list(range(self.begin_y+20, self.end_y-15))

                    if any(val in self.pillar_x for val in self.flappy_x) and any(val in self.pillar_y for val in self.flappy_y):
                        self.finish = True

                if self.pillar_point_list[i] is False and self.finish is False and self.start:
                    if self.flappy_coords:
                        if (self.flappy_coords[1][0] - 80) > self.pv_val:
                            self.pillar_point_list[i] = True
                            self.score += 1
                
                if self.mask_pillar_list[i] is not None:
                    self.mask_pillar = self.mask_pillar_list[i]

                self.rois = self.frame[0:self.h, self.pv_val: self.p_w+self.pv_val]
                self.rois[np.where(self.mask_pillar)] = 0
                self.rois += pillar

                self.pillar_list[i] = pillar
                self.mask_pillar_list[i] = self.mask_pillar

                if self.finish is False and self.start:
                    self.pv_list[i] -= 10

                if self.pv_list[i] <= 0: 
                    self.pv_list[i] = 525
                    self.pillar_resize[i] = 0

                    self.mask_pillar_list[i] = None
                    self.pillar_list[i] = None

                    self.pillar_point_list[i] = False

            if self.face.detections:
                detection = self.face.detections[0]
                self.face_data = detection.location_data
                self.nose_tip = self.face_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value]

                if self.nose_tip and self.finish is False:
                    self.nose_tip_x = round(self.nose_tip.x * self.w)
                    self.nose_tip_y = round(self.nose_tip.y * self.h)

                    if not self.w - self.flappy_size + 200 > self.nose_tip_x > 200: self.nose_tip_x = self.prev_nose_tip_x
                    if not self.h - self.flappy_size + 55 > self.nose_tip_y > 55: self.nose_tip_y = self.prev_nose_tip_y

                if self.prev_flappy is not None:
                    self.flappy = self.originial_flappy

                    if self.angle > self.dest_angle: self.angle -= 5
                    elif self.angle < self.dest_angle: self.angle += 5

                    if abs(self.prev_nose_tip_x - self.nose_tip_x) > 3 or abs(self.prev_nose_tip_y - self.nose_tip_y) > 3:
                        self.dest_angle = await self.calculate_angle([self.nose_tip_x, self.nose_tip_y], self.prev_flappy, [self.prev_nose_tip_x, self.prev_nose_tip_y]) * 10
                        if self.dest_angle > 45: self.dest_angle = 45
                        elif self.dest_angle < -45 : self.dest_angle = -45
                    else:
                        self.dest_angle = 0

                    self.flappy = await self.rotate_image(self.flappy, self.angle)
                    self.mask_flappy = await self.mask(self.flappy)
            
            self.txt = 'SCORE: '
            cv2.putText(self.frame, self.txt, (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, False)
            cv2.putText(self.frame, str(self.score), (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, False)

            if self.nose_tip_x is not None and self.nose_tip_y is not None:
                self.prev_nose_tip_x = self.nose_tip_x
                self.prev_nose_tip_y = self.nose_tip_y
                self.prev_flappy = [self.prev_nose_tip_x-100, self.prev_nose_tip_y-25]

                self.flappy_coords = [[self.nose_tip_x-200, self.nose_tip_y-55], [self.nose_tip_x-200+self.flappy_size, self.nose_tip_y-55+self.flappy_size]]

                self.roi = self.frame[self.nose_tip_y-55: self.nose_tip_y-55+self.flappy_size, self.nose_tip_x-200: self.nose_tip_x-200+self.flappy_size]

                self.roi[np.where(self.mask_flappy)] = 0
                self.roi += self.flappy

            self.buffer_img = await self.loop.run_in_executor(None, cv2.imencode, '.jpg', self.frame)
            self.b64_img = base64.b64encode(self.buffer_img[1]).decode('utf-8')

            await self.send(self.b64_img)
        else:
            await self.default()
            await self.close()