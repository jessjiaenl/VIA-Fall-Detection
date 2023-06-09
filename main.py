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

def render(result):
    # GUI here
    print(result)
    return 1

def predictNRenderVid(model, vid_path):
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    while ret:
        result = predict(model, frame)
        render(result)
        ret, frame = cap.read()


if __name__ == '__main__':
    model = Fall_Detection.FallDet()

    # useFallDet, model_path, useVid, vid_path = sys.argv
    useFallDet, model_path, useVid, vid_path = True, "", True, "./datasets/model2_vids/resized_jess_pickup.MOV"

    if not useFallDet: model = Unit_Test.SingleModel(model_path)

    if useVid:
        predictNRenderVid(model, vid_path)
    else:
        quit = False
        while(not quit):
            pic = read_from_cam()
            result = predict(model, pic)
            render(result)



