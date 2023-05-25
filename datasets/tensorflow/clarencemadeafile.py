import os

import numpy as np

import tensorflow as tf
#assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

#image_path = tf.keras.utils.get_file('photos')
 #             'flower_photos.tgz',
  #                  'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   #                       extract=True)
#image_path = os.path.join(os.path.dirname(image_path), 'photos')
image_path = os.path.join(os.getcwd(), 'photos')
data = DataLoader.from_folder(image_path)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)
plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(image.numpy(), cmap=plt.cm.gray)
      plt.xlabel(data.index_to_label[label.numpy()])
plt.show()

model = image_classifier.create(train_data, validation_data=validation_data)
model.summary()
loss, accuracy = model.evaluate(test_data)

# A helper function that returns 'red'/'black' depending on if its two input
# parameter matches or not.
def get_label_color(val1, val2):
      if val1 == val2:
          return 'black'
      else:
          return 'red'

# Then plot 100 test images and their predicted labels
# If a prediction result is different from the label provided label in "test"
# dataset, we will highlight it in red color.
plt.figure(figsize=(20, 20))
predicts = model.predict_top_k(test_data)
for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take(100)):
    ax = plt.subplot(10, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image.numpy(), cmap=plt.cm.gray)
    predict_label = predicts[i][0][0]
    color = get_label_color(predict_label,
    test_data.index_to_label[label.numpy()])
    ax.xaxis.label.set_color(color)
    plt.xlabel('Predicted: %s' % predict_label)
plt.show()
model.export(export_dir='.')
model.export(export_dir='.', export_format=ExportFormat.LABEL)
model.evaluate_tflite('model.tflite', test_data)
