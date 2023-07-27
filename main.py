import old_Fall_Detection
import Fall_Detection
import diffdiff_Fall_Detection
import eight_Fall_Detection
import Single_Model
import GUI
import oldGUI
import cv2
import numpy as np
import time
import subprocess

# Calls the predict function 
def predict(model, frame):
    return model.predictFrame(frame)

# Print out the board result
def render(frame, result, performance, fpstime):
    print(result)
    print()
    UI.draw_frame(frame, result, performance, fpstime) 

def predictNRenderVid(modelcpy, vid_path, vid_path2):
    # Video capture for webcam
    cap = cv2.VideoCapture(int(ID), cv2.CAP_V4L2)

    # Video capture from video
    cap2 = cv2.VideoCapture(vid_path2)
    
    # Main Loop 
    frame = None
    fpstime = 0
    while True:
        # Changes the model if you switched tabs
        if UI.switched == True:
            if UI.currtab == 0: 
                modelcpy = eight_Fall_Detection.FallDet()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif UI.currtab == 1:
                modelcpy = Single_Model.SingleModel(model_paths[UI.currtab], [[1,368,368,3]], [[1,46,46,57]], np.uint8, np.uint8)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                modelcpy = Single_Model.SingleModel(model_paths[UI.currtab], [[1,300,300,3]], [[1,1917,21],[1,1917,4]], np.uint8, np.float)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read() 
        start1 = time.time()
        result, performance = predict(modelcpy, frame) 
        end1 = time.time()
        print("BOARD TIME")
        print(end1-start1)
        start = time.time()
        render(frame, result, performance, fpstime) 
        # Checks key event
        if UI.check_key():
            break
        end = time.time()
        print("FULL TIME")
        print(end-start)
        print("Exited via key press")

if __name__ == '__main__':
    ### edit vid paths, model paths, which model to start with here ###
    fall_vid, pose_vid = "./../s-still-fall2.mp4", "./../skate.mp4"
    model_paths = ["", "./../dla_models/openpose_mobilenetv0.75_quant_1x368x368x3.dla", "./../dla_models/mobilenet_ssd_pascal_quant.dla"]
    modelidx = 0
    ###################################################################
    
    UI = oldGUI.oldGUI() # initilize UI
    model = None
    if modelidx == 0: # fall detection
        model = eight_Fall_Detection.FallDet()
    elif modelidx == 1: # openpose
        model = Single_Model.SingleModel(model_paths[modelidx], [[1,368,368,3]], [[1,46,46,57]], np.uint8, np.uint8) 
    else: # object detection
        model = Single_Model.SingleModel(model_paths[modelidx], [[1,300,300,3]], [[1,1917,21],[1,1917,4]], np.uint8, np.float) 
    ID = 0
    # Clear the holder file in case ran many times
    command_clean = "cp /dev/null cmd_for_id_detector.txt"
    subprocess.run(command_clean, shell=True)

    # Generate new ID
    command_run = "ls -1 /dev/video* | cut -c11- | sort -n | tail -2 | head -1 >> cmd_for_id_detector.txt"
    subprocess.run(command_run, shell=True)

    # Read into variable
    with open("cmd_for_id_detector.txt", "r") as file:
        ID = file.read()
    predictNRenderVid(model, fall_vid, pose_vid)