import numpy as np
import cv2
import tensorflow as tf
  
def cropFrameToSquare(frame):
    h, w, _ = frame.shape
    target_len = min(h,w)
    start_x, start_y = (w - target_len)//2, (h - target_len)//2
    return frame[start_y:start_y+target_len, start_x:start_x+target_len, :]

def predict(interpreter, frame, input_shape, input_type, input_index, output_index):
    # match model input shape
    frame = cropFrameToSquare(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
    frame_rgb = cv2.resize(frame_rgb, (input_shape[1], input_shape[2]), interpolation=cv2.INTER_AREA) # resize frame to 224x224
    frame_rgb = np.expand_dims(frame_rgb, axis=0) # resize to match tensor size [224x224x3] -> [1x224x224x3]
    # match model input type
    frame_rgb = frame_rgb.astype(input_type)

    # predict using interpreter
    interpreter.set_tensor(input_index, frame_rgb)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_index)
    output_data = output_data[0]

    output_probs = np.exp(output_data.astype(float))/np.sum(np.exp(output_data.astype(float))) # without tf
    predicted_index = np.argmax(output_data)
    class_labels = ["Moving", "Still"]
    predicted_class = class_labels[predicted_index]

    prob = np.around(max(output_probs), decimals = 2) # without tf
    if predicted_class == "Still": return 1-prob
    else: return prob

def predictNRenderVid(interpreter, vid_path, input_shape, input_type, input_index, output_index):
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    arr = []
    while ret:
        result = predict(interpreter, frame, input_shape, input_type, input_index, output_index)
        arr.append(result)
        print(result)
        ret, frame = cap.read()
    print(arr)

if __name__ == "__main__":
    # initialize tensor for model1
    interpreter = tf.lite.Interpreter(model_path="./tflite_models/model.tflite")
    interpreter.allocate_tensors()
    output = interpreter.get_output_details()
    input = interpreter.get_input_details()
    output_index = output[0]['index']
    input_index = input[0]['index']

    input_shape = [1,224,224,3]
    output_shape = [1,2]
    input_type = np.uint8

    predictNRenderVid(interpreter, "./datasets/model2_vids/original/s-walk1.mp4", 
                    input_shape, input_type, input_index, output_index)
