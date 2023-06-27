import sys
import Fall_Detection
import Single_Model
import cv2
import numpy as np

def read_from_cam():
  cap = cv2.VideoCapture(0)
  ret, frame = cap.read()
  return frame # frame is reshaped in model not here

def predict(model, frame):
  return model.predictFrame(frame)

def render(result):
  print(result)

def predictNRenderVid(model, vid_path):
  cap = cv2.VideoCapture(vid_path)
  ret, frame = cap.read()
  while True:
    if not ret:
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      ret, frame = cap.read()
    result = predict(model, frame)
    render(result)
    ret, frame = cap.read()

if __name__ == '__main__':
  '''
  modelidx {0 : fall detection, 1 : open pose, 2 : object detection}
  '''
  model_paths = ["", "./tflite_models/openpose_mobilenetv0.75_quant_1x368x368x3.tflite", "./tflite_models/mobilenet_ssd_pascal_quant.tflite"]
  # modelidx, useVid, vid_path = sys.argv
  modelidx, useVid, vid_path = 2, True, "./datasets/model1_vids/original/jess_IMG_0480.MOV"
  model = None
  if modelidx == 0: model = Fall_Detection.FallDet()
  elif modelidx == 1: model = Single_Model.SingleModel(model_paths[modelidx], [[1,368,368,3]], [[1,46,46,57]], np.uint8, np.uint8)
  else: model = Single_Model.SingleModel(model_paths[modelidx], [[1,300,300,3]], [[1,1917,21],[1,1917,4]], np.uint8, np.uint8)

  if useVid:
      predictNRenderVid(model, vid_path)
  else:
      quit = False
      while(not quit):
          pic = read_from_cam()
          result = predict(model, pic)
          render(result)