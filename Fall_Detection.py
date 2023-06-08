import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')
import tensorflow_datasets as tfds

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
import datetime

import matplotlib.pyplot as plt

import cv2
import sys
sys.path.append("./compiler_and_runtime/Neuropl")

import neuropl

class FallDet:
    interpreter = None
    output_index = 0
    input_index = 0

    model1 = None
    model2 = None

    probs = []
    frame = None
    frame_rgb = None

    threshold = 0.88

    def __init__(self):
        '''
        # initialize tensor for model1
        self.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()
        output = self.interpreter.get_output_details()
        input = self.interpreter.get_input_details()
        self.output_index = output[0]['index']
        self.input_index = input[0]['index']
        # load model2
        self.model2 = tf.keras.models.load_model("model2")
        '''
        # using neuropl API
        self.model1 = neuropl.Neuropl("model1.dla") # model1 in: uint8 (1x224x224x3) out: uint8 (1x2)
        self.model2 = neuropl.Neuropl("model2.dla") # model2 in: uint8 (1x16) out: uint8 (1x1)
    
    def updateFrame(self, frame): # call this everytime we get a frame
        self.frame = frame
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_rgb = np.expand_dims(self.frame_rgb, axis=0) # match shape to model
    
    def predictFrame(self): # call this after updateFrame
        '''
        self.interpreter.set_tensor(self.input_index, self.frame_rgb)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_index)
        output_data = output_data[0]
        '''
        # using neuropl
        output_data = self.model1.predict(self.frame_rgb) # assume this outputs [movingprob, stillprob]

        # below remains the same regardless of neuropl API usage
        output_probs = tf.nn.softmax(output_data.astype(float))
        predicted_index = np.argmax(output_data)
        class_labels = ["Moving", "Still"]
        predicted_class = class_labels[predicted_index]

        prob = np.around(max(output_probs.numpy()), decimals = 2)
        if predicted_class == "Still": self.probs += [1-prob]
        else: self.probs += [1-prob]        

        result = False # default = not falling
        if len(self.probs) == 16:
            result = self.predictVid()
            self.probs = self.probs[1:]

        return result
    
    def predictVid(self): #  main doesn't call this function
        model2_in = np.array(self.probs).reshape((1, 16))
        vid_preds = self.model2.predict(model2_in) # uint8 1x1
        return (vid_preds.reshape((1, len(vid_preds))) > self.threshold)[0][0]