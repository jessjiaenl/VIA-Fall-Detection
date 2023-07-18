import old_Fall_Detection
import Single_Model
import GUI
import cv2
import numpy as np

def predict(model, frame):
    return model.predictFrame(frame)

def render(frame, result, performance, fpstime):
    print(result)
    print()
    UI.draw_frame(frame, result, performance, fpstime) 

def predictNRenderVid(modelcpy, vid_path, vid_path2):
    # Fall Detection
    cap = cv2.VideoCapture(vid_path)
    # Open Pose
    cap2 = cv2.VideoCapture(vid_path2)
    # Object Detection
    cap3 = cv2.VideoCapture(126, cv2.CAP_V4L2) # 126th index is for camera, don't change 
    
    frame = None
    fpstime = 0

    # Main Loop 
    while True:
        if UI.switched == True:
            if UI.currtab == 0: 
                modelcpy = old_Fall_Detection.FallDet()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif UI.currtab == 1:
                modelcpy = Single_Model.SingleModel(model_paths[UI.currtab], [[1,368,368,3]], [[1,46,46,57]], np.uint8, np.uint8)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                modelcpy = Single_Model.SingleModel(model_paths[UI.currtab], [[1,300,300,3]], [[1,1917,21],[1,1917,4]], np.uint8, np.float)
                cap3.set(cv2.CAP_PROP_POS_FRAMES, 0)
            UI.switched = False
        
        if UI.currtab == 0: # Fall Detection
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        elif UI.currtab == 1: # Open Pose
            ret, frame = cap2.read()
            if not ret:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap2.read() 
        else: # Object Detection
            ret, frame = cap3.read()
            if not ret:
                cap3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap3.read() 

        result, performance = predict(modelcpy, frame) 
        render(frame, result, performance, fpstime) 

        # Checks if you quit the screen (press 'q')
        if UI.check_key():
            break
            print("Exited via key press")

if __name__ == '__main__':
    ### edit vid paths, model paths, which model to start with here ###
    fall_vid, pose_vid = "./../s-still-fall2.mp4", "./../demo_pose_1.MOV"
    model_paths = ["", "./../dla_models/openpose_mobilenetv0.75_quant_1x368x368x3.dla", "./../dla_models/mobilenet_ssd_pascal_quant.dla"]
    modelidx = 0
    ###################################################################
    
    UI = GUI.GUI() # initilize UI
    model = None
    if modelidx == 0: # fall detection
        model = old_Fall_Detection.FallDet()
    elif modelidx == 1: # openpose
        model = Single_Model.SingleModel(model_paths[modelidx], [[1,368,368,3]], [[1,46,46,57]], np.uint8, np.uint8) 
    else: # object detection
        model = Single_Model.SingleModel(model_paths[modelidx], [[1,300,300,3]], [[1,1917,21],[1,1917,4]], np.uint8, np.float) 
    
    predictNRenderVid(model, fall_vid, pose_vid)