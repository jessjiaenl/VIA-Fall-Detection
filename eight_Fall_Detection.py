import numpy as np

import tensorflow as tf
# assert tf.__version__.startswith('2')

# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras import layers

import cv2

import sys
sys.path.append("./compiler_and_runtime/Neuropl")
# import neuropl

class FallDet:
    interpreter = None
    output_index = 0
    input_index = 0

    model1 = None
    model2 = None

    # hard coded parameters for fall detection
    model1_input_shape = [1,224,224,3] 
    model1_output_shape = [1,2]
    model2_input_shape = [1,8]
    model2_output_shape = [1,1]
    input_type = np.uint8

    probs = []

    threshold = 200

    prev_frame_gray = None
    is_first_frame = True

    def __init__(self):
        
        # initialize tensor for model1
        self.interpreter = tf.lite.Interpreter(model_path="./tflite_models/new_m1.tflite") # "./tflite_models/model.tflite"
        self.interpreter.allocate_tensors()
        output = self.interpreter.get_output_details()
        input = self.interpreter.get_input_details()
        self.output_index = output[0]['index']
        self.input_index = input[0]['index']
        # load model2
        self.model2 = tf.keras.models.load_model("new_m2")
        '''
        # using neuropl API
        self.model1 = neuropl.Neuropl("model1.dla") # model1 in: uint8 (1x224x224x3) out: uint8 (1x2)
        self.model2 = neuropl.Neuropl("model2.dla") # model2 in: uint8 (1x16) out: uint8 (1x1)
        '''
        self.probs = []
        self.prev_frame_gray = None
        self.is_first_frame = True
        diff_threshold = 0.2

    def cropFrameToSquare(self, frame):
        h, w, _ = frame.shape
        target_len = min(h,w)
        start_x, start_y = (w - target_len)//2, (h - target_len)//2
        return frame[start_y:start_y+target_len, start_x:start_x+target_len, :]
    
    def predictFrame(self, frame):
        # match model input shape
        frame = self.cropFrameToSquare(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
        frame_rgb = cv2.resize(frame_rgb, (self.model1_input_shape[1], self.model1_input_shape[2]), interpolation=cv2.INTER_AREA) # resize frame to 224x224
        frame_rgb = np.expand_dims(frame_rgb, axis=0) # resize to match tensor size [224x224x3] -> [1x224x224x3]
        # match model input type
        frame_rgb = frame_rgb.astype(self.input_type)

        # for extra filter
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.is_first_frame:
            self.prev_frame_gray = curr_frame_gray
            self.is_first_frame = False
        frame_diff = cv2.absdiff(self.prev_frame_gray, curr_frame_gray)
        frame_avg_diff = np.sum(frame_diff) / (self.model1_input_shape[1]*self.model1_input_shape[2])

        
        # predict using interpreter
        self.interpreter.set_tensor(self.input_index, frame_rgb)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_index)
        output_data = output_data[0]
        '''
        # predict using neuropl
        output_data = self.model1.predict(frame_rgb)[0] # API outputs [[movingprob, stillprob]]
        '''

        # below remains the same regardless of neuropl API usage
        # output_probs = tf.nn.softmax(output_data.astype(float)) # use tf
        output_probs = np.exp(output_data.astype(float))/np.sum(np.exp(output_data.astype(float))) # without tf
        predicted_index = np.argmax(output_data)
        class_labels = ["Moving", "Still"]
        predicted_class = class_labels[predicted_index]

        # prob = np.around(max(output_probs.numpy()), decimals = 2) # use tf
        prob = np.around(max(output_probs), decimals = 2) # without tf
        if predicted_class == "Still": self.probs += [1-prob]
        elif frame_avg_diff < self.diff_threshold: self.probs += [1-prob] # small value
        else: self.probs += [prob]

        result = False # default = not falling
        if len(self.probs) == 8:
            result = self.predictVid()
            self.probs = self.probs[1:]

        return result
    
    def predictVid(self): #  main doesn't call this function
        model2_in = np.array(self.probs).reshape((1, 8))
        vid_preds = self.model2.predict(model2_in)[0] # uint8 1x1
        vid_preds = vid_preds*255
        return (vid_preds.reshape((1, len(vid_preds))) > self.threshold) # [[bool]]
        # return vid_preds.reshape((1, len(vid_preds)))