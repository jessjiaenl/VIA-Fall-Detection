import sys
sys.path.append("./compiler_and_runtime/Neuropl")
import neuropl
import cv2
import numpy as np

class SingleModel:
  model = None
  input_type = np.uint8 # manually specified by client
  output_type = np.uint8 # manually specified by client
  input_shape = [1,224,224,3]

  def __init__(self, model_path):
    self.model = neuropl.Neuropl(model_path) # .dla
    # self.input_type = self.model.get_intput_type()
    # self.output_type = self.model.get_output_type()
  
  def predictFrame(self, frame):
    # match model input shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
    frame_rgb = cv2.resize(frame_rgb, (self.input_shape[1], self.input_shape[2]), interpolation=cv2.INTER_AREA) # resize image dim
    frame_rgb = np.expand_dims(frame_rgb, axis=0) # resize to match tensor size

    # match model input type
    input = frame_rgb.astype(self.input_type)

    return self.model.predict(input) # assume this outputs [movingprob, stillprob]