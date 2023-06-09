import sys
sys.path.append("./compiler_and_runtime/Neuropl")
import neuropl
import cv2
import numpy as np

blank_image = np.zeros((224,224), np.uint8)

class model:
  model = None
  input_type = int
  output_type = int

  def __init__(self, model_path):
      self.model = neuropl.Neuropl(model_path) # .dla
      self.input_type = self.model.get_intput_type()
      self.output_type = self.model.get_output_type()
  
  def predictFrame(self, frame):
    # match model input shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.expand_dims(self.frame_rgb, axis=0) 
    # match model input type
    input = frame_rgb

    return self.model.predict(input) # assume this outputs [movingprob, stillprob]