from __future__ import division, absolute_import
import re
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os.path import isfile, join
import random
import sys


class EMR:
    def __init__(self):
        self.target_classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    def build_network(self):
        print("[LOG] iniciando red")
        self.network = input_data(shape=[None, 48, 48, 1])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        self.network = dropout(self.network, 0.3)
        self.network = fully_connected(self.network, 3072, activation='relu')
        self.network = fully_connected(self.network, len(self.target_classes), activation='softmax')
        self.network = regression(self.network,
                                  optimizer='momentum',
                                  loss='categorical_crossentropy')
        self.model = tflearn.DNN(
            self.network,
            checkpoint_path='model_1_nimish',
            max_checkpoints=1,
            tensorboard_verbose=2
        )
        self.load_model()

    def predict(self, image):
        if image is None:
            return None
        image = image.reshape([-1, 48, 48, 1])
        return self.model.predict(image)

    def load_model(self):
        if isfile("models/model_1_nimish.tflearn.meta"):
            self.model.load("models/model_1_nimish.tflearn")
            print('[INFO] cargando modelo')
        else:
            print("[ERROR] cargando modelo")


if __name__ == "__main__":
    network = EMR()

