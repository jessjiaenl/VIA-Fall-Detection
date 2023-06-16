import sys
#import Fall_Detection
import GUI
#import Unit_Test
import cv2
import numpy as np
#import tensorflow as tf
import time



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

def get_video_dimensions(video_path):
    try:
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Read the first frame
        ret, frame = video.read()

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        return width, height

    except Exception as e:
        print("An error occurred: ", e)

    finally:
        # Release the video capture object
        video.release()

def crop_frame(frame, target_width, target_height):
    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Calculate the starting point for cropping
    start_x = (width - target_width) // 2
    start_y = (height - target_height) // 2

    # Perform the cropping
    cropped_frame = frame[start_y:start_y+target_height, start_x:start_x+target_width, :]

    return cropped_frame
'''
def pose_predict(image):
    OUTPUT_RATIO = 0.008926701731979847
    OUTPUT_BIAS = 126
    threshold = 0.2/OUTPUT_RATIO+OUTPUT_BIAS
    # Example usage
    # video_path = "posevid.mp4"
    # width, height = get_video_dimensions(video_path)
    # cropped_frame = crop_frame(image, height, height)
    resized_frame = cv2.resize(image, (368, 368))
    color_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(color_frame, axis=0)
    frame = frame.astype(np.uint8)
    
    

    # set input -> invoke -> access output
    interpreter.set_tensor(input_index, frame)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)
    return output_data
'''
if __name__ == '__main__':
    '''interpreter stuff'''
    # interpreter = tf.lite.Interpreter(model_path="./openpose_mobilenetv0.75_quant_1x368x368x3.tflite")
    # interpreter.allocate_tensors()

    # output = interpreter.get_output_details()
    # input = interpreter.get_input_details()
    # output_index = output[0]['index']
    # input_index = input[0]['index']





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
    vid_path = "vid4.mp4"
    vid_path2 = "posevid.mp4"
    cap = cv2.VideoCapture(vid_path)
    cap2 = cv2.VideoCapture(vid_path2)
    # for i in range(0,2):
    #     ret, frame = cap.read()
    #     width, height = get_video_dimensions(vid_path2)
    #     cropped_frame = crop_frame(frame, height, height)
    #     frame = cropped_frame
    #     start_time = time.time()
    #     result = pose_predict(frame)
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     print(f"Execution time: {execution_time} seconds")
    #     #print(result)
    # Specify the file path and name
    file_path = "./pose_output.txt"

    # Initialize the larger list
    larger_list = []

    # Read the contents of the file
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Strip any leading/trailing whitespaces and newline characters
            line = line.strip()
            line = line.replace(", ",",")
            # Split the line by spaces to separate the elements
            elements_str = line.split()
            
            # Ensure the line has exactly 18 elements
            if len(elements_str) == 18:
                # Convert elements to tuples
                elements = [tuple(map(int, elem.strip('()').split(','))) for elem in elements_str]
                
                # Append the elements list to the larger list
                larger_list.append(elements)
    
    new_large = []

    for row in larger_list:
        new_row = []
        for pt in row:
            if pt[0] != -1:
                newx = (int)(pt[1]/46*900)
                newy = (int)(pt[0]/46*900)
                pt = (newx,newy) 
                new_row.append(pt)
            else:
                new_row.append((-1,-1))
        new_large.append(new_row)
    row = 0
    while True:
        if UI.currtab >= 1:
            if UI.switched == True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                UI.switched = False
                row = 0
            ret, frame = cap2.read()
            width, height = get_video_dimensions(vid_path2) 
            cropped_frame = crop_frame(frame, height, height)
            frame = cropped_frame
            result = new_large[row]
            row +=1
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
        

    # if useVid:
    #     predictNRenderVid(model, vid_path)
    # else:
    #     quit = False
    #     while(not quit):
    #         pic = read_from_cam()
    #         result = predict(model, pic)
    #         render(result)




