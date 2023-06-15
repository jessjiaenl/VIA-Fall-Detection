import sys
#import Fall_Detection
import GUI
#import Unit_Test
import cv2
import numpy as np

def read_from_cam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # frame is reshaped in model not here
    return frame
'''
def predict(model, frame):
    return model.predictFrame(frame)
'''
def render(result, modelidx):
    # UI.draw_tab(pic) 
    print(result, modelidx)
    return 1
'''
def predictNRenderVid(model, vid_path):
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    while True:
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        result = predict(model, frame)
        render(result, modelidx)
        ret, frame = cap.read()
'''

if __name__ == '__main__':
    # modelidx, model_path, useVid, vid_path = sys.argv
    UI = GUI.GUI()
    '''
    modelidx, model_path, useVid, vid_path = 0, "", True, "jess_IMG_0480.MOV"
    model = Fall_Detection.FallDet()
    if modelidx != 0: model = Unit_Test.SingleModel(model_path)
    '''
    '''
    modelidx is {0 : fall detection, 1 : open pose, 2 : object detection}
    some test vid paths:
    "./datasets/model1_vids/original/jess_IMG_0480.MOV"
    "./datasets/model1_vids/resized_IMG_0480.MOV"
    model paths:
    open pose: "openpose_mobilenetv0.75_quant_1x368x368x3.dla"
    obj det: "mobilenet_ssd_pascal_quant.dla"
    '''
    result = 0
    vid_path = "_IMG_2610.mp4"
    vid_path2 = "vid1.mp4"
    cap = cv2.VideoCapture(vid_path)
    cap2 = cv2.VideoCapture(vid_path2)
    ret, frame = cap.read()
    
    while True:
        
        if UI.currtab >= 1:
            if UI.switched == True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                UI.switched = False
            ret, frame = cap2.read()
            #frame = read_from_cam
        else:
            if UI.switched == True:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                UI.switched = False
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        
        UI.draw_frame(frame, result)
        if UI.check_key() == True:
            break
        
    #print("test")
        


    # if useVid:
    #     predictNRenderVid(model, vid_path)
    # else:
    #     quit = False
    #     while(not quit):
    #         pic = read_from_cam()
    #         result = predict(model, pic)
    #         render(result)




