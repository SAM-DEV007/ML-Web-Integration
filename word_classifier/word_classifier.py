from django.conf import settings

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import os


txt_arg = ["Not Available (N/A)", "Saying Hi", "Borrow Money", "YouTube Interaction"]

model_path = str(settings.BASE_DIR / 'word_classifier/Model/WordClassifier_Model.h5')
if not os.path.exists(model_path):
    if not os.path.isdir(str(settings.BASE_DIR / 'word_classifier/Model')):
        os.mkdir(str(settings.BASE_DIR / 'word_classifier/Model'))
    
    from word_classifier import download_model

# model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False) # Uncomment this line if using sufficient memory for the model to load (saves model prediction time)

def predict(txt: str) -> str:
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False) # Comment this line if the above same line is uncommented
    result_arg = (np.squeeze(model.predict([txt])))
    result = np.argmax(result_arg)

    return f'{txt_arg[result]} - ({np.round(result_arg[result]*100, 2)}%)'