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
  input_shape = [[1,300,300,3]] # manually specified by client
  output_shape = [[1,1917,21], [1,1917,4]] # manually specified by client

  # for tflite interpreter usage
  interpreter = None
  output_indices = [0]
  input_index = 0 # assumes single input

  def __init__(self, modelPath, input_shape, output_shape, input_type, output_type):
    # self.model = neuropl.Neuropl(modelPath) # .dla
    self.input_shape, self.output_shape = input_shape, output_shape
    self.input_type, self.output_type = input_type, output_type
    # initialize tensor
    self.interpreter = tf.lite.Interpreter(model_path=modelPath)
    self.interpreter.allocate_tensors()
    output = self.interpreter.get_output_details()
    input = self.interpreter.get_input_details()
    self.output_indices = [output[i]['index'] for i in range(len(output))]
    self.input_index = input[0]['index'] # assumes single input

  def cropFrameToSquare(self, frame):
    h, w, _ = frame.shape
    target_len = min(h,w)
    start_x, start_y = (w - target_len)//2, (h - target_len)//2
    return frame[start_y:start_y+target_len, start_x:start_x+target_len, :]

  def predictFrame(self, frame):
    # match model input shape
    frame = self.cropFrameToSquare(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
    frame_rgb = cv2.resize(frame_rgb, (self.input_shape[0][1], self.input_shape[0][2]), interpolation=cv2.INTER_AREA) # resize image dim
    frame_rgb = np.expand_dims(frame_rgb, axis=0) # resize to match tensor size

    # match model input type
    input = frame_rgb.astype(self.input_type)

    # predict using interpreter
    self.interpreter.set_tensor(self.input_index, input)
    self.interpreter.invoke()
    output_data = [self.interpreter.get_tensor(self.output_indices[i]) for i in range(len(self.output_indices))]
    # list of ndarray cuz outputs have different dimensions

    # predict using neuropl
    # output_data = self.model.predict(input)

    return output_data