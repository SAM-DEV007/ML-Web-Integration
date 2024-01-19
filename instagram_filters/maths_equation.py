from django.conf import settings

from urllib.request import urlopen
from instagram_filters import storage

import random
import cv2
import os
import time
import numpy as np


def maths_gen(localauth):
    obj = MathsEquation()

    while True:
        frame = obj.main(localauth)
        if frame is not None:
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else: yield frame


class MathsEquation():
    def __init__(self):
        '''
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        '''

        self.hide = False
        self.get_dir = False
        self.question_asked = False
        self.broadcast_corr = False
        self.broadcast_incorr = False
        self.last = False
        self.end = False

        self.first_launch = True

        self.first_launch_time = time.time()
        self.curr = time.time()

        self.face_cascade_path = str(settings.BASE_DIR / 'instagram_filters/MathsEquation_Data/frontface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)

        self.dict_eq = self.generate()
        self.d_ques = list(self.dict_eq.keys())
        self.d_ans = list(self.dict_eq.values())

        self.left_op, self.right_op = '', ''

        self.x, self.y, self.w, self.h = 0, 0, 0, 0
        self.prev_left, self.prev_right = 0, 0
        self.ans, self.ans_corr = 0, 0
        self.direction = 0
        self.debounce = 0
        self.end_time = 0
        self.correct_op_direction = 0


    def __del__(self):
        cv2.destroyAllWindows()


    def generate(self) -> dict:
        '''Generates dictionary containing the maths equation with answer'''

        sign = ['+', '-', '*', '/']
        placeholder = ['+', '-', 'x', '/']

        dict_eq = dict()

        for _ in range(5):
            eq = ''
            random_step = random.randint(1, 2)
            n = 30

            for j in range(random_step+1):
                num = random.randint(1, n)
                if eq:
                    r_sign = random.sample(sign, k=1)[0]
                    r_num = random.randint(1, n - (j*5))

                    eq += f' {str(r_sign)}'
                    eq += f' {str(r_num)}'
                    continue
                eq += str(num)

            pl = ''
            for t in eq:
                if t in sign:
                    pl += placeholder[sign.index(t)]
                    continue
                pl += t
            
            dict_eq[pl] = round(eval(eq), 2)
        
        return dict_eq


    def get_direction(self, prev_left: int, prev_right: int, x: int, w: int) -> str:
        '''Returns the direction of the face'''

        if 0 not in (prev_left, prev_right):
            if prev_right < (x + w):
                if (x + w) - prev_right > 6:
                    return 1
            elif prev_left > x:
                if prev_left - x > 6:
                    return -1
        return 0


    def generate_options(self, curr_ques: str, curr_ans: int):
        '''Generates the correct and incorrect option'''

        correct = curr_ans
        if '-' in curr_ques: incorrect = -curr_ans
        else:
            random_no = random.randint(1, 5)
            incorrect = random.sample([correct + random_no, correct - random_no], k=1)[0]
        
        return str(correct), str(incorrect)


    def put_text_options(self, frame, x: int, y: int, w: int, direction: int, left_op: int, right_op: int):
        '''Adds options in the Quiz Area'''

        cv2.putText(frame, left_op, (x - w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, right_op, (x + (2 * w) - (18 * len(right_op)), y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


    def put_text(self, frame, txt: str, x: int, y: int, w: int, h: int, color: tuple):
        '''Add the text in the Quiz Area'''

        t_x = (((x - w) + (x + (2 * w))) // 2)
        t_y = (((y + 20) + (y - h + 20)) // 2)
        cv2.putText(frame, txt, (t_x - (len(txt) * 9), t_y), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2, cv2.LINE_AA)


    def add_quiz_area(self, frame, x: int, y: int, w: int, h: int, color_val: list, first_launch: bool, first_launch_time, d_ques: list, d_ans: list, ans: int, ans_corr: int, direction: int, question_asked: bool, left_op: str, right_op: str, correct_op_direction: int, broadcast_corr: bool, broadcast_incorr: bool, last: bool, end: bool):
        '''Add quiz box near the forehead'''

        get_dir = False

        if not broadcast_corr and not broadcast_incorr: cv2.rectangle(frame, (x - w, y + 20), (x + (2 * w), y - h + 20), color_val[0], -1)
        elif broadcast_corr:
            cv2.rectangle(frame, (x - w, y + 20), (x + (2 * w), y - h + 20), color_val[2], -1)
            broadcast_corr = False
        elif broadcast_incorr:
            cv2.rectangle(frame, (x - w, y + 20), (x + (2 * w), y - h + 20), color_val[3], -1)
            broadcast_incorr = False

        cv2.rectangle(frame, (x - w, y + 20), (x + (2 * w), y - h + 20), (255, 255, 255))

        if (time.time() - first_launch_time) > 5:
            first_launch = False

        if first_launch:
            self.put_text(frame, 'Welcome to Maths Quiz!', x, y, w, h, color_val[1])
        elif end:
            self.put_text(frame, f'Correct answers: {ans_corr}', x, y, w, h, color_val[1])
        else:
            if last:
                curr_ques = d_ques[ans-1]
                curr_ans = d_ans[ans-1]
            else:
                curr_ques = d_ques[ans]
                curr_ans = d_ans[ans]

            get_dir = True

            if not question_asked and not last:
                question_asked = True

                correct_op_direction = random.sample([-1, 1], k=1)[0]

                if correct_op_direction == -1: left_op, right_op = self.generate_options(curr_ques, curr_ans)
                else: right_op, left_op = self.generate_options(curr_ques, curr_ans)

            self.put_text(frame, curr_ques, x, y, w, h, color_val[1])
            self.put_text_options(frame, x, y, w, direction, left_op, right_op)

        return first_launch, ans, ans_corr, get_dir, left_op, right_op, question_asked, correct_op_direction, broadcast_corr, broadcast_incorr


    def main(self, localauth):
        '''The main function for starting the program'''

        if not storage.IMG_PATH.get(localauth): 
            return

        if self.ans > (len(self.dict_eq) - 1):
            if not self.last: self.last = True
            else:
                self.end = True

                if not self.end_time:
                    self.end_time = time.time()
                if time.time() - self.end_time > 5:
                    pass

        #_, self.frame = self.vid.read()
        req = urlopen(storage.IMG_PATH[localauth])
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

        self.frame = cv2.cvtColor(cv2.imdecode(arr, -1), cv2.COLOR_BGRA2BGR)
        self.frame = cv2.flip(self.frame, 1)

        if self.frame.shape[0] != 480 or self.frame.shape[1] != 640: 
            return

        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.face = self.face_cascade.detectMultiScale(self.gray, 1.1, 5)

        self.prev_left, self.prev_right = self.x, self.x + self.w

        self.color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.correct_color = (0, 255, 0)
        self.incorrect_color = (0, 0, 255)

        self.color_val = [self.color, self.text_color, self.correct_color, self.incorrect_color]

        if type(self.face) == tuple:
            if not self.hide:
                self.first_launch, self.ans, self.ans_corr, self.get_dir, self.left_op, self.right_op, self.question_asked, self.correct_op_direction, self.broadcast_corr, self.broadcast_incorr = self.add_quiz_area(self.frame, self.x, self.y, self.w, self.h, self.color_val, self.first_launch, self.first_launch_time, self.d_ques, self.d_ans, self.ans, self.ans_corr, self.direction, self.question_asked, self.left_op, self.right_op, self.correct_op_direction, self.broadcast_corr, self.broadcast_incorr, self.last, self.end)

                self.prev = time.time()
                if (self.prev - self.curr) > 1:
                    self.hide = True
        else:
            if self.hide: self.hide = False
            else:
                (self.x, self.y, self.w, self.h) = self.face[0]
                self.first_launch, self.ans, self.ans_corr, self.get_dir, self.left_op, self.right_op, self.question_asked, self.correct_op_direction, self.broadcast_corr, self.broadcast_incorr = self.add_quiz_area(self.frame, self.x, self.y, self.w, self.h, self.color_val, self.first_launch, self.first_launch_time, self.d_ques, self.d_ans, self.ans, self.ans_corr, self.direction, self.question_asked, self.left_op, self.right_op, self.correct_op_direction, self.broadcast_corr, self.broadcast_incorr, self.last, self.end)

                if self.get_dir:
                    self.direction = self.get_direction(self.prev_left, self.prev_right, self.x, self.w)
                    self.get_dir = False
                
                if self.direction and self.question_asked:
                    if not self.debounce: 
                        self.debounce = time.time()
                    if time.time() - self.debounce > 1:
                        self.debounce = 0
                        
                        if self.direction == self.correct_op_direction:
                            self.ans_corr += 1
                            self.broadcast_corr = True
                        else: self.broadcast_incorr = True

                        self.direction = 0
                        self.ans += 1
                        self.question_asked = False

                self.curr = time.time()
        
        _, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()