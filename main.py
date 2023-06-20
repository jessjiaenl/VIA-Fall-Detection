import sys
import Fall_Detection
# import GUI
import Unit_Test
import cv2
import numpy as np

def read_from_cam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # frame is reshaped in model not here
    return frame

def predict(model, frame):
    return model.predictFrame(frame)

def processObjDet(result): # outputs bounding 
    confidences, boxes = result[0][0], result[1][0]
    row_sums = confidences.sum(axis=1)
    confidences = confidences/row_sums[:, np.newaxis] # normalize confidence
    threshold = 0.15
    rem_box_class = []
    for i in range(len(confidences)): # preserve boxes that detected something & what they detected
        if max(confidences[i]) >= threshold: rem_box_class.append((boxes[i], np.argmax(confidences[i])))
    return rem_box_class

def render(result, modelidx):
    # UI.draw_tab(pic)
    print(result)
    return 1

def predictNRenderVid(model, vid_path):
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    while True:
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        result = predict(model, frame)
        if modelidx == 2: result = processObjDet(result)
        render(result, modelidx)
        ret, frame = cap.read()


if __name__ == '__main__':
    # modelidx, model_path, useVid, vid_path = sys.argv
    # UI = GUI.GUI()
    modelidx, model_path, useVid, vid_path = 2, "./tflite_models/mobilenet_ssd_pascal_quant.tflite", True, "./datasets/model1_vids/original/jess_IMG_0480.MOV"
    model = None
    if modelidx == 0: model = Fall_Detection.FallDet()
    else: model = Unit_Test.SingleModel(model_path)
    '''
    modelidx is {0 : fall detection, 1 : open pose, 2 : object detection}
    some test vid paths:
    "./datasets/model1_vids/original/jess_IMG_0480.MOV"
    "./datasets/model1_vids/resized_IMG_0480.MOV"
    model paths:
    open pose: "openpose_mobilenetv0.75_quant_1x368x368x3.dla"
    obj det: "./tflite_models/mobilenet_ssd_pascal_quant.dla"
    '''

    if useVid:
        predictNRenderVid(model, vid_path)
    else:
        quit = False
        while(not quit):
            pic = read_from_cam()
            result = predict(model, pic)
            render(result)



