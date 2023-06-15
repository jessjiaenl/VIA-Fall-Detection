import numpy as np

# import tensorflow as tf
# assert tf.__version__.startswith('2')

# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras import layers

import cv2

import sys
sys.path.append("./compiler_and_runtime/Neuropl")
import neuropl

class FallDet:
    interpreter = None
    output_index = 0
    input_index = 0

    model1 = None
    model2 = None

    model1_input_shape = [1,224,224,3] # hard coded for fall detection
    model1_output_shape = [1,2] # hard coded for fall detection
    model2_input_shape = [1,16] # hard coded for fall detection
    model2_output_shape = [1,1]  # hard coded for fall detection
    input_type = np.uint8 # hard coded for fall detection

    probs = []
    frame = None
    frame_rgb = None

    threshold = 0.88

    def __init__(self):
        '''
        # initialize tensor for model1
        self.interpreter = tf.lite.Interpreter(model_path="./tflite_models/model.tflite")
        self.interpreter.allocate_tensors()
        output = self.interpreter.get_output_details()
        input = self.interpreter.get_input_details()
        self.output_index = output[0]['index']
        self.input_index = input[0]['index']
        # load model2
        self.model2 = tf.keras.models.load_model("model2")
        '''
        # using neuropl API
        self.model1 = neuropl.Neuropl("model1.dla") # model1 in: uint8 (1x224x224x3) out: uint8 (1x2)
        self.model2 = neuropl.Neuropl("model2.dla") # model2 in: uint8 (1x16) out: uint8 (1x1)
        
    
    def predictFrame(self, frame):
        # match model input shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
        frame_rgb = cv2.resize(frame_rgb, (self.model1_input_shape[1], self.model1_input_shape[2]), interpolation=cv2.INTER_AREA) # resize frame to 224x224
        frame_rgb = np.expand_dims(frame_rgb, axis=0) # resize to match tensor size [224x224x3] -> [1x224x224x3]
        # match model input type
        frame_rgb = frame_rgb.astype(self.input_type)

        '''
        # predict using interpreter
        self.interpreter.set_tensor(self.input_index, frame_rgb)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_index)
        output_data = output_data[0]
        '''
        # predict using neuropl
        output_data = self.model1.predict(frame_rgb, len(self.model1_input_shape), len(self.model1_output_shape)) # assume this outputs [movingprob, stillprob]


        # below remains the same regardless of neuropl API usage
        # output_probs = tf.nn.softmax(output_data.astype(float)) # use tf
        output_probs = np.exp(output_data.astype(float))/np.sum(np.exp(output_data.astype(float))) # without tf
        predicted_index = np.argmax(output_data)
        class_labels = ["Moving", "Still"]
        predicted_class = class_labels[predicted_index]

        # prob = np.around(max(output_probs.numpy()), decimals = 2) # use tf
        prob = np.around(max(output_probs), decimals = 2) # without tf
        if predicted_class == "Still": self.probs += [1-prob]
        else: self.probs += [1-prob]        

        result = False # default = not falling
        if len(self.probs) == 16:
            result = self.predictVid()
            self.probs = self.probs[1:]

        return result
    
    def predictVid(self): #  main doesn't call this function
        model2_in = np.array(self.probs).reshape((1, 16))
        vid_preds = self.model2.predict(model2_in, len(self.model2_input_shape), len(self.model2_output_shape)) # uint8 1x1
        return (vid_preds.reshape((1, len(vid_preds))) > self.threshold)[0][0]