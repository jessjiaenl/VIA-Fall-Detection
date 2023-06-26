import sys
import Fall_Detection
import GUI
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

def processObjDet(result): # outputs list of (bbox info, class id of bbox) for good bboxes
    confidences, boxes = result[0][0], result[1][0]
    row_sums = confidences.sum(axis=1)
    confidences = confidences/row_sums[:, np.newaxis] # normalize confidence
    threshold = 0.15
    rem_box_class = []
    for i in range(len(confidences)): # preserve boxes that detected something & what they detected
        if max(confidences[i]) >= threshold: rem_box_class.append((boxes[i], np.argmax(confidences[i])))
    return rem_box_class

def render(result):
    # UI.draw_tab(pic)
    print(result)
    return 1

def predictNRenderVid(model, vid_path, vid_path2):
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    cap2 = cv2.VideoCapture(vid_path2)
    while True:
        if UI.currtab >= 1:
            if UI.switched == True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                UI.switched = False
            ret, frame = cap2.read()
            if not ret:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap2.read()
            result = predict(model, frame)
        else:
            if UI.switched == True:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                UI.switched = False
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            result = predict(model, frame)
        UI.draw_frame(frame, result)
        if UI.check_key() == True:
            break
    print("Exited via key press")

    '''
    while True:
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        result = predict(model, frame)
        #if modelidx == 2: result = processObjDet(result)
        render(result)

        ret, frame = cap.read()'''


if __name__ == '__main__':
    UI = GUI.GUI()
    '''
    modelidx {0 : fall detection, 1 : open pose, 2 : object detection}
    '''
    model_paths = ["", "./tflite_models/openpose_mobilenetv0.75_quant_1x368x368x3.tflite", "./tflite_models/mobilenet_ssd_pascal_quant.tflite"]
    # modelidx, useVid, vid_path = sys.argv
    modelidx, useVid, vid_path, vid_path2 = 2, True, "./datasets/model1_vids/original/jess_IMG_0480.MOV", "./posevid.mp4"
    model = None
    if modelidx == 0: model = Fall_Detection.FallDet()
    elif modelidx == 1: model = Unit_Test.SingleModel(model_paths[modelidx], [[1,368,368,3]], [[1,46,46,57]], np.uint8, np.uint8)
    else: model = Unit_Test.SingleModel(model_paths[modelidx], [[1,300,300,3]], [[1,1917,21],[1,1917,4]], np.uint8, np.uint8)
    
    if useVid:
        predictNRenderVid(model, vid_path, vid_path2)
    else:
        quit = False
        while(not quit):
            pic = read_from_cam()
            result = predict(model, pic)
            render(result)



