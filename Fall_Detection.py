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

frame_classifications = []
moving_prob = []
arguments = sys.argv
frame = arguments[0]
frame_classifications.append(frame)

''' Formulate Input Data (frame_rgb) '''
# Convert the frame to RGB format
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# Make frame input data and ensure its type matches the model
frame_rgb = np.expand_dims(frame_rgb, axis=0)

''' Classify the Frame '''
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# get_output_details() and get_input_details() return list of dictionaries of tensor details
# keys: name, index, shape, shape_signature, dtype, quantization, ...
# len(input) = len(output) = 1, so access the first element
output = interpreter.get_output_details()
input = interpreter.get_input_details()
output_index = output[0]['index']
input_index = input[0]['index']

# set input -> invoke -> access output
interpreter.set_tensor(input_index, frame_rgb)
interpreter.invoke()

output_data = interpreter.get_tensor(output_index)
# If the output_data shape is (batch_size, num_classes), select the first frame
output_data = output_data[0]

# Convert each entry into probability
output_probs = tf.nn.softmax(output_data.astype(float))

# Find the index of the highest probability
predicted_index = np.argmax(output_data)

# Assuming you have a list of class labels corresponding to the model's output classes
class_labels = ["Moving", "Still"]

# Get the predicted class label
predicted_class = class_labels[predicted_index]

# Print the predicted class label
# print("Predicted Class:", predicted_class)
frame_classifications.append((predicted_class, max(output_probs.numpy())))

prob = np.around(max(output_probs.numpy()), decimals = 2)
if predicted_class == "Still":
    
    moving_prob.append(np.subtract(1, prob))
else:
    moving_prob.append(prob)

if len(moving_prob) == 16:
    # Model 2 dataset preparation
    moving_probs_trimmed = moving_prob[:-(len(moving_prob)%16)]
    model2_in = np.array(moving_probs_trimmed).reshape((len(moving_prob)//16, 16))

    # Model 2 prediction
    model2 = tf.keras.models.load_model("model2")
    vid_preds = model2.predict(model2_in)
    threshold = 0.9
    bools = vid_preds.reshape((1, len(vid_preds))) > threshold
    print(bools)
    print(vid_preds.reshape((1, len(vid_preds))))

