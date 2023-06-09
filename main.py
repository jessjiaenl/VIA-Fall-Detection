import sys
import Fall_Detection
import Unit_Test
import cv2

def read_from_cam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # reshape frame then return frame
    return 0

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
    model = Fall_Detection.FallDet()

    # modelidx, model_path, useVid, vid_path = sys.argv
    modelidx, model_path, useVid, vid_path = 0, "", True, "./datasets/model1_vids/original/jess_IMG_0480.MOV"
    if modelidx != 0: model = Unit_Test.SingleModel(model_path)
    '''
    modelidx {0 : fall detection, 1 : open pose, 2 : object detection}
    '''
    if useVid:
        predictNRenderVid(model, vid_path)
    else:
        quit = False
        while(not quit):
            pic = read_from_cam()
            result = predict(model, pic)
            render(result)



