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
    model2 = None

    # hard coded parameters for fall detection
    img_shape = (224,224)
    model2_input_shape = [1,16]
    model2_output_shape = [1,1]
    input_type = np.uint8

    diffs = []
    prev_diff = None
    prev_frame_gray = None
    is_first_frame, is_second_frame = True, True

    threshold = 25.5

    def __init__(self):
        # load model2
        self.model2 = tf.keras.models.load_model("new_model2")
        '''
        # using neuropl API
        self.model2 = neuropl.Neuropl("model2.dla") # model2 in: uint8 (1x16) out: uint8 (1x1)
        '''

    def cropFrameToSquare(self, frame):
        h, w, _ = frame.shape
        target_len = min(h,w)
        start_x, start_y = (w - target_len)//2, (h - target_len)//2
        return frame[start_y:start_y+target_len, start_x:start_x+target_len, :]
    
    def predictFrame(self, curr_frame):
        # match model input shape
        curr_frame = self.cropFrameToSquare(curr_frame)
        cv2.resize(curr_frame, self.img_shape) # scale to 224*224
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if self.is_first_frame:
            self.prev_frame_gray = curr_frame_gray
            self.prev_diff = curr_frame_gray - self.prev_frame_gray
            self.is_first_frame = False
        elif self.is_second_frame:
            self.prev_diff = curr_frame_gray - self.prev_frame_gray
            self.is_second_frame = False
        frame_diff = curr_frame_gray - self.prev_frame_gray
        diffdiff = frame_diff - self.prev_diff
        frame_avg_diff = np.sum(np.abs(diffdiff)) / (self.img_shape[0]*self.img_shape[1])
        # match model input type
        frame_avg_diff = frame_avg_diff.astype(self.input_type)
        # print(frame_avg_diff)


        self.diffs.append(frame_avg_diff)
        self.prev_frame_gray = curr_frame_gray
        self.prev_diff = frame_diff

        result = False # default = not falling
        if len(self.diffs) == 16:
            result = self.predictVid()
            self.diffs = self.diffs[1:]
        return result
    
    def predictVid(self): #  main doesn't call this function
        model2_in = np.array(self.diffs).reshape((1, 16)) / 255
        print(model2_in)
        vid_preds = self.model2.predict(model2_in) # dtype=uint8 (1,1)
        vid_preds = vid_preds*255 # quantize
        return (vid_preds > self.threshold) # [[bool]]
        # return vid_preds