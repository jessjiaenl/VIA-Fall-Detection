import subprocess
import Fall_Detection

def read_from_cam():
    return 0

def predict(fd, frame):
    fd.updateFrame(frame)
    fd.predictFrame()

def render(result):
    #GUI here
    return 1



if __name__ == '__main__':
    fd = Fall_Detection.FallDet()

    quit = False
    while(not quit):
        pic = read_from_cam()
        result = predict(fd, pic)
        render(result)



