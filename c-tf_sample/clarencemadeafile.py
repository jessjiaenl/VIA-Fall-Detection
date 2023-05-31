import os
import cv2
import cv2
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
'''
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
'''
for i in range(0,20):
    print("done classifing stuff")
model.export(export_dir='.')
model.export(export_dir='.', export_format=ExportFormat.LABEL)

model.evaluate_tflite('model.tflite', test_data)
'''
#test
# Function to classify a single frame
def classify_frame(frame, model):
    preprocessed_frame = preprocess_image(frame)
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    label = np.argmax(prediction)
    return label

# Video file path
video_path = "./../vid2.mp4"

# Load the video using OpenCV
cap = cv2.VideoCapture(video_path)

# Initialize an empty list to store the frame classifications
frame_classifications = []

# Loop through the frames of the video
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Classify the frame
    label = classify_frame(frame_rgb, model)

    # Add the frame classification to the list
    frame_classifications.append(label)

    # Display the frame
    cv2.imshow("Video", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Convert the frame classifications to a numpy array
frame_classifications = np.array(frame_classifications)

# Print the frame classifications


print("Frame Classifications:", frame_classifications)
'''