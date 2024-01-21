from django.conf import settings
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import requests
import sys
import traceback


model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Model\\Model_Data', 'WordClassifier_Model.h5')

if not os.path.exists(model_path):
    download_model('1LAiyCV0p6v-lROdXbtrzKlF4APNmM3Qm', model_path)

model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
txt = input("Enter your message: ")

result_arg = (np.squeeze(model.predict([txt])))
result = np.argmax(result_arg)
txt_arg = ["N/A", "Saying Hi", "Borrow Money", "YouTube Interaction"]

print(f"\nThe prediction is based between - {' | '.join(txt_arg)}.\n")
print(f"The prediction is {txt_arg[result]}, with the probability of {np.round(result_arg[result]*100, 2)}%")