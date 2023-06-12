import sys
import Fall_Detection
import Unit_Test
import cv2

def read_from_cam():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    # frame is reshaped in model not here
    return frame

def predict(model, frame):
    return model.predictFrame(frame)

def render(result, modelidx):
    # GUI here
    print(result, modelidx)
    return 1

def predictNRenderVid(model, vid_path):
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    while ret:
        result = predict(model, frame)
        render(result, modelidx)
        ret, frame = cap.read()


if __name__ == '__main__':
    # modelidx, model_path, useVid, vid_path = sys.argv
    modelidx, model_path, useVid, vid_path = 0, "", True, "./datasets/model1_vids/original/jess_IMG_0480.MOV"
    model = Fall_Detection.FallDet()
    if modelidx != 0: model = Unit_Test.SingleModel(model_path)
    '''
    modelidx is {0 : fall detection, 1 : open pose, 2 : object detection}
    some test vid paths:
    "./datasets/model1_vids/original/jess_IMG_0480.MOV"
    "./datasets/model1_vids/resized_IMG_0480.MOV"
    '''

    if useVid:
        predictNRenderVid(model, vid_path)
    else:
        quit = False
        while(not quit):
            pic = read_from_cam()
            result = predict(model, pic)
            render(result)



