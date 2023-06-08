import hello_ext
import boostClass
import neuropl
import cv2 
import numpy as np

blank_image = np.zeros((224,224), np.uint8)

#print(hello_ext.greet())
# t = boostClass.Test("Hello")
# t.scream()
# print(t.myMax(12,15))
model = neuropl.Neuropl("PLEASEPLEASEPLEASE")
model.print_attributes()
model.setModelPath("HEHEHEHEHEHEHEHEHEHEH")
model.print_attributes()
model.predict(blank_image)

#model.predict(blank_image)