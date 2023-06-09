import sys
import neuropl
import cv2
import numpy as np

blank_image = np.zeros((224,224), np.uint8)

def predict(model_path, input_path):
  model = neuropl.Neuropl(model_path)

  # reformat input so that it matches model input type
  model.get_input_type()
  input = input_path

  return model.predict(input)

class model:
    model = None
    input_type = int
    output_type = int
    frame = None

    def __init__(self, model_path):
        self.model = neuropl.Neuropl(model_path) # .dla
        self.input_type = self.model.get_intput_type()
        self.output_type = self.model.get_output_type()
    
    def predictFrame(self, frame): # call this after updateFrame
        self.frame = frame
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_rgb = np.expand_dims(self.frame_rgb, axis=0) # match shape to model

        self.interpreter.set_tensor(self.input_index, self.frame_rgb)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_index)
        output_data = output_data[0]
        '''
        # using neuropl
        output_data = self.model1.predict(self.frame_rgb) # assume this outputs [movingprob, stillprob]
        '''

        # below remains the same regardless of neuropl API usage
        output_probs = tf.nn.softmax(output_data.astype(float))
        predicted_index = np.argmax(output_data)
        class_labels = ["Moving", "Still"]
        predicted_class = class_labels[predicted_index]

        prob = np.around(max(output_probs.numpy()), decimals = 2)
        if predicted_class == "Still": self.probs += [1-prob]
        else: self.probs += [1-prob]        

        result = False # default = not falling
        if len(self.probs) == 16:
            result = self.predictVid()
            self.probs = self.probs[1:]

        return result
    
    def predictVid(self): #  main doesn't call this function
        model2_in = np.array(self.probs).reshape((1, 16))
        vid_preds = self.model2.predict(model2_in) # uint8 1x1
        return (vid_preds.reshape((1, len(vid_preds))) > self.threshold)[0][0]