import subprocess
import Fall_Detection
import cv2

def read_from_cam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    return 0

def predict(fd, frame):
    fd.updateFrame(frame)
    fd.predictFrame()

def render(result):
    #GUI here
    return 1

def predictNRenderVid(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    while ret:
        predict(fd, frame)
        result = predict(fd, pic)
        render(result)

        ret, frame = cap.read()


if __name__ == '__main__':
    fd = Fall_Detection.FallDet()

    # take argument or built in?
    useVid, path = True, ""

    if useVid:
        predictNRenderVid(path)
    else:
        quit = False
        while(not quit):
            pic = read_from_cam()
            result = predict(fd, pic)
            render(result)



