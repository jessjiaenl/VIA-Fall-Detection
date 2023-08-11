import numpy as np
import cv2
import sys
sys.path.append("./../Neuropl")
import neuropl
import time

class FallDet:
    model1_input_shape = [1,224,224,4]  # Added 4th channel
    model1_output_shape = [1,1]
    model2_input_shape = [1,8]
    model2_output_shape = [1,1]
    input_type = np.uint8
    # sylvia's
    # threshold = 210
    # kevin's
    threshold = 49

    model1 = None
    model2 = None
    probs = []
    prev_frame_gray = None
    is_first_frame = True
    
    def __init__(self):
        self.model1 = neuropl.Model("./../dla_models/k_m1_quant.dla")
        self.model2 = neuropl.Model("./../dla_models/k_m2_quant.dla")
        self.probs = []

    def compute_diff_channel(self, curr_frame_gray):
        if self.is_first_frame:
            diff_channel = np.zeros_like(curr_frame_gray)
            self.prev_frame_gray = curr_frame_gray
            self.is_first_frame = False
        else:
            diff_channel = cv2.absdiff(self.prev_frame_gray, curr_frame_gray)
            self.prev_frame_gray = curr_frame_gray
        return diff_channel

    def cropFrameToSquare(self, frame):
        h, w, _ = frame.shape
        target_len = min(h,w)
        start_x, start_y = (w - target_len)//2, (h - target_len)//2
        return frame[start_y:start_y+target_len, start_x:start_x+target_len, :]

    def predictFrame(self, frame):
        start = time.time()
        frame = self.cropFrameToSquare(frame)
        frame = cv2.resize(frame, (self.model1_input_shape[1], self.model1_input_shape[2]), interpolation=cv2.INTER_AREA) 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_channel = self.compute_diff_channel(curr_frame_gray)
        frame_rgbx = np.concatenate((frame_rgb, diff_channel[:,:,np.newaxis]), axis=-1)  # Adding 4th channel

        frame_rgbx = np.expand_dims(frame_rgbx, axis=0)
        frame_rgbx = frame_rgbx.astype(self.input_type)
        # print('frame_rgbx shape', frame_rgbx.shape)
        # print('frame_rgbx type', frame_rgbx.dtype)
        output_data = self.model1.predict(frame_rgbx)[0][0] # Now getting single value
        # print('output_data type before: ', output_data.dtype)
        self.probs.append(output_data/255.0) # cast to float32 for model2 input
        # print('output_data type after: ', output_data.dtype)
        result = False 
        if len(self.probs) == 8: 
            result = self.predictVid()
            self.probs = self.probs[1:]

        timediff = time.time() - start
        return result, timediff
    
    def predictVid(self):
        model2_in = np.array(self.probs).reshape((1, 8))
        vid_preds = self.model2.predict(model2_in)[0]
        # self.model2.print_profiled_qos_data()
        print(f"\nm2 predicted val = {vid_preds[0]}")
        return (vid_preds.reshape((1, len(vid_preds))) > self.threshold)[0][0]
    
    
