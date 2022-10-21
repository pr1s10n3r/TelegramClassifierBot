import logging

import cv2
import tensorflow as tf
from tensorflow import keras


class ImageClassifier:
    def __init__(self, imgpath: str):
        self.imgpath = imgpath
        self.classes = ['airplane', 'automobile', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def load_classifier_model(self, model_filepath: str) -> None:
        self.model = tf.keras.models.load_model(model_filepath)

    def load_resized_image(self) -> None:
        img = cv2.imread(self.imgpath, cv2.IMREAD_COLOR)
        resized = cv2.resize(img, (32, 32))
        self.reshaped = resized.reshape(-1, 32, 32, 3)

    def predict(self) -> str:
        prediction = self.model.predict(self.reshaped)
        plist = prediction[0].tolist()
        logging.info(plist)
        return self.classes[plist.index(max(plist))]

    def to_spanish(self, text: str) -> str:
        translations = {
            'airplane': 'avión',
            'automobile': 'automóvil',
            'bird': 'pájaro',
            'cat': 'gato',
            'deer': 'venado',
            'dog': 'perro',
            'frog': 'sapo',
            'horse': 'caballo',
            'ship': 'barco',
            'truck': 'camión'
        }

        return translations[text]
