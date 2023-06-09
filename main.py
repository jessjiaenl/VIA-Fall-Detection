import sys
import Fall_Detection
import cv2

def read_from_cam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # reshape frame then return frame
    return 0

def predict(fd, frame):
    return fd.predictFrame(frame)

def render(result):
    # GUI here
    print(result)
    return 1

def predictNRenderVid(fd, path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    while ret:
        result = predict(fd, frame)
        render(result)

        ret, frame = cap.read()


if __name__ == '__main__':
    fd = Fall_Detection.FallDet()

    # useVid, path = sys.argv
    useVid, path = True, "./datasets/model2_vids/resized_jess_pickup.MOV"

    if useVid:
        predictNRenderVid(fd, path)
    else:
        quit = False
        while(not quit):
            pic = read_from_cam()
            result = predict(fd, pic)
            render(result)



