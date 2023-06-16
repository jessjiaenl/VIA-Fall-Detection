import sys
sys.path.append("./compiler_and_runtime/Neuropl")
import neuropl
import cv2
import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

class SingleModel:
  model = None
  input_type = np.uint8 # manually specified by client
  output_type = np.uint8 # manually specified by client
  input_shape = [1,300,300,3] # manually specified by client
  output_shape = [1,1917,4] # manually specified by client

  # for tflite interpreter usage
  interpreter = None
  output_index = 0
  input_index = 0

  def __init__(self, modelPath):
    # self.model = neuropl.Neuropl(modelPath) # .dla
    # initialize tensor
    self.interpreter = tf.lite.Interpreter(model_path=modelPath)
    self.interpreter.allocate_tensors()
    output = self.interpreter.get_output_details()
    input = self.interpreter.get_input_details()
    self.output_index = output[0]['index']
    self.input_index = input[0]['index']

  
  def predictFrame(self, frame):
    # match model input shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
    frame_rgb = cv2.resize(frame_rgb, (self.input_shape[1], self.input_shape[2]), interpolation=cv2.INTER_AREA) # resize image dim
    frame_rgb = np.expand_dims(frame_rgb, axis=0) # resize to match tensor size

    # match model input type
    input = frame_rgb.astype(self.input_type)

    # predict using interpreter
    self.interpreter.set_tensor(self.input_index, input)
    self.interpreter.invoke()

    output_data = self.interpreter.get_tensor(self.output_index)
    # output_data = output_data[0]

    # predict using neuropl
    # output_data = self.model.predict(input, len(self.input_shape), len(self.output_shape))

    # print(output_data.shape)
    # for i in range(output_data.shape[1]):
    #   confidence = output_data[0][i]

    return output_data