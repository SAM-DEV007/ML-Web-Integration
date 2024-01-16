from django.conf import settings

import mediapipe as mp
import numpy as np
import tensorflow as tf

import cv2
import copy
import itertools
import os
import time


def flappy_gen():
    obj = HandGesture()

    while True:
        frame = obj.main()
        yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


class predict():
    def __init__(self, model_path = settings.BASE_DIR / 'instagram_filters/HandGesture_Data/Model.tflite', num_threads = 1):
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


class HandGesture():
    def __init__(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        show_palm = True
        show_ok = True
        show_peace = True

        img_busy = False

        model_db = 0
        gesture = None

        mphands = mp.solutions.hands
        hands = mphands.Hands()

        model_predict = predict()

        mask_big_img = None
        big_img = None
        big_img_time = None

        mask_palm_img_holder = None
        mask_ok_img_holder = None
        mask_peace_img_holder = None

        palm_img_holder = None
        ok_img_holder = None
        peace_img_holder = None

        temp_mask_palm_img_holder = None
        temp_mask_ok_img_holder = None
        temp_mask_peace_img_holder = None

        temp_palm_img_holder = None
        temp_ok_img_holder = None
        temp_peace_img_holder = None

        img_size = 150

        palm_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Data\\', 'Palm.png')
        ok_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Data\\', 'Ok.png')
        peace_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Data\\', 'Peace.png')

        size = 100

        palm = cv2.imread(palm_path)
        palm = cv2.resize(palm, (size, size))
        mask_palm = mask(palm)
        
        ok = cv2.imread(ok_path)
        ok = cv2.resize(ok, (size, size))
        mask_ok = mask(ok)

        peace = cv2.imread(peace_path)
        peace = cv2.resize(peace, (size, size))
        mask_peace = mask(peace)

    
    def __del__(self):
        cv2.destroyAllWindows()


    def landmark_list(image, landmarks):
        '''Generates the landmark list'''

        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return pre_process_data(landmark_point)


    def pre_process_data(landmark_list):
        '''Pre processes the data for the trained model'''

        temp_landmark_list = copy.deepcopy(landmark_list)

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


    def detect_hands(frame, hands, model_predict):
        '''Detects the hand and returns the suitable gesture'''

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_landmarks = result.multi_hand_landmarks
        gesture = None

        if hand_landmarks:
            for handLMs in hand_landmarks:
                if handLMs:
                    landmark = landmark_list(frame, handLMs)
                    gesture = model_predict(landmark)
        
        if gesture == 3: gesture = None
        return gesture


    def mask(img):
        '''Masks the image'''

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

        return mask


    def pre_img(frame, size, img, mask_img, gesture):
        '''Initializes the hand img to the frame'''

        img_section = [-600, -360, -130]
        img_x = img_section[gesture]

        roi = frame[-465: -465+size, img_x: img_x+size]
    
        roi[np.where(mask_img)] = 0
        roi += img


    def get_img(frame):
        '''Gets the images after the gesture'''

        cv2.rectangle(frame, [130, 170], [510, 450], (0, 0, 0), 2)

        cv2.rectangle(frame, [110, 150], [530, 170], (255, 255, 255), -1)
        cv2.rectangle(frame, [110, 450], [530, 470], (255, 255, 255), -1)
        cv2.rectangle(frame, [110, 170], [130, 450], (255, 255, 255), -1)
        cv2.rectangle(frame, [510, 170], [530, 450], (255, 255, 255), -1)

        cv2.rectangle(frame, [110, 150], [530, 470], (0, 0, 0), 2)
            
        img = frame[149: 472, 109: 532] 

        return img


    def hold_img_main(frame, img, mask_img):
        '''Holds the img on the screen for a while'''

        roi = frame[149: 472, 109: 532]
        roi[np.where(mask_img)] = 0

        roi += img


    def change_img(frame, img, mask_img, gesture: int, img_size: int):
        '''Changes the gesture image'''

        img_section = [-600, -400, -200]
        img_x = img_section[gesture]

        roi = frame[-465: -465+img_size, img_x: img_x+img_size]
        roi[np.where(mask_img)] = 0

        roi += img


    def main():
        '''The main function to start the filter'''

        _, frame = cam.read() 
        frame = cv2.flip(frame, 1) 

        if show_palm: pre_img(frame, size, palm, mask_palm, 0)
        else: change_img(frame, palm_img_holder, mask_palm_img_holder, 0, img_size)
        if show_ok: pre_img(frame, size, ok, mask_ok, 1)
        else: change_img(frame, ok_img_holder, mask_ok_img_holder, 1, img_size)
        if show_peace: pre_img(frame, size, peace, mask_peace, 2)
        else: change_img(frame, peace_img_holder, mask_peace_img_holder, 2, img_size)

        if big_img is not None:
            if (time.time() - big_img_time) > 1.5:
                
                img_busy = False
                model_db = 0
                big_img_time = None
                big_img = None
                mask_big_img = None
            else:
                hold_img_main(frame, big_img, mask_big_img) 
        
        if big_img is None:
            if temp_palm_img_holder is not None:
                palm_img_holder = temp_palm_img_holder
                mask_palm_img_holder = temp_mask_palm_img_holder

                temp_mask_palm_img_holder = None
                temp_palm_img_holder = None
                show_palm = False

            if temp_ok_img_holder is not None:
                ok_img_holder = temp_ok_img_holder
                mask_ok_img_holder = temp_mask_ok_img_holder
                
                temp_mask_ok_img_holder = None
                temp_ok_img_holder = None
                show_ok = False

            if temp_peace_img_holder is not None:
                peace_img_holder = temp_peace_img_holder
                mask_peace_img_holder = temp_mask_peace_img_holder

                temp_mask_peace_img_holder = None
                temp_peace_img_holder = None
                show_peace = False
        
        if not model_db: model_db = time.time()
        
        if not img_busy:
            gesture = detect_hands(frame[165: 480, 0: 640], hands, model_predict)

        if gesture is not None and (time.time() - model_db) > 1 and not img_busy:
            img_busy = True 
            temp_frame = copy.deepcopy(frame) 

            img = get_img(temp_frame) 
            img = cv2.detailEnhance(img, sigma_r=0.15, sigma_s=3) 
            
            m = mask(img)
            hold_img_main(frame, img, m)

            big_img_time = time.time()
            big_img = img
            mask_big_img = m
            
            img = cv2.resize(img, (img_size, img_size)) 
            if gesture == 0: 
                temp_palm_img_holder = img
                temp_mask_palm_img_holder = mask(img)
            elif gesture == 1: 
                temp_ok_img_holder = img
                temp_mask_ok_img_holder = mask(img)
            elif gesture == 2:
                temp_peace_img_holder = img
                temp_mask_peace_img_holder = mask(img)
            gesture = None

        _, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()