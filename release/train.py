import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader
from tensorflow.keras.callbacks import TensorBoard
import cv2
import datetime
import os

def trainM1(data_path, train_data_ratio):
  data = DataLoader.from_folder(data_path)
  train_data, test_data = data.split(train_data_ratio)
  m1 = image_classifier.create(train_data, use_augmentation=True)
  loss, acc = m1.evaluate(test_data)
  return m1, loss, acc

def trainM2(default_data, falling_data):
  # create dataset
  inputs = np.concatenate((default_data, falling_data))
  outputs = np.concatenate((np.zeros(len(default_data)), np.ones(len(falling_data)))) # ones are falling , zeros are default
  dataset_size = len(inputs)
  new_indices = np.random.permutation(dataset_size) # shuffle indices to shuffle X and y at the same time
  inputs, outputs = inputs[new_indices], outputs[new_indices]

  train_size = int(0.8*dataset_size)
  test_size = dataset_size - train_size

  X_train, y_train = inputs[:train_size], outputs[:train_size] #x = images, y = label 
  X_test, y_test = inputs[train_size:], outputs[train_size:]

  # data generator
  def shuffle_generator(image, label, seed):
    idx = np.arange(len(image))
    np.random.default_rng(seed).shuffle(idx)
    for i in idx: yield image[i], label[i]
  
  train_data = tf.data.Dataset.from_generator(
    shuffle_generator,
    args=[X_train, y_train.reshape(len(y_train), 1), 42],
    output_signature=(tf.TensorSpec(shape=(8,), dtype=tf.uint8),
                            tf.TensorSpec(shape=(1,), dtype=tf.uint8))).batch(128)
  
  # create model
  m2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(1, activation='sigmoid')])
  m2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # tensorboard
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath="./model_checkpoint/",
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)
  
  # train
  m2.fit(train_data, epochs = 10, callbacks = [tensorboard_callback, early_stop_callback, model_checkpoint_callback])

  # evaluate
  loss, acc = m2.evaluate(X_test, y_test)
  return m2, loss, acc

def batchPredict(vid_path, m1_path):
    cap = cv2.VideoCapture(vid_path)
    probs = []
    while True:
        ret, frame = cap.read() 
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.expand_dims(frame_rgb, axis=0)

        interpreter = tf.lite.Interpreter(model_path="./tflite_models/new_m1.tflite")
        interpreter.allocate_tensors()
        output = interpreter.get_output_details()
        input = interpreter.get_input_details()
        output_index = output[0]['index']
        input_index = input[0]['index']

        interpreter.set_tensor(input_index, frame_rgb)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)
        output_data = output_data[0]
        output_probs = tf.nn.softmax(output_data.astype(float))

        predicted_index = np.argmax(output_data)
        class_labels = ["Moving", "Still"]
        predicted_class = class_labels[predicted_index]        
        prob = np.around(max(output_probs.numpy()), decimals = 2)

        if predicted_class == "Still": probs.append(np.subtract(1, prob))
        else: probs.append(prob)
    return probs

def batchPredictMultVids(vids_dir, m1_path):
  record = []
  for vid in os.listdir(vids_dir):
    vid_path = os.path.join(vids_dir, vid)
    vid_probs = batchPredict(vid_path, m1_path)
    print(vid_path)
    for i in range(len(vid_probs)-7):
      record.append(vid_probs[i:i+8])
      for j in range(i, i+8):
        print(vid_probs[j], end=" ")
      print()
    print()
  return record

def augmentM2Data(default_data, falling_data):
  def limit(x):
    if x > 1: return 1 - (x-1)
    elif x < 0: return abs(x)
    return x
  newLimit = np.vectorize(limit)

  def augment(less_data, copy_count):
    mu, sig = 0, 0.005
    rowi = 0
    for i in range(copy_count):
      err = np.random.normal(mu, sig, 8)
      scale = 1
      row = less_data[rowi]*scale + err
      row = newLimit(row)
      less_data = np.vstack([less_data, row])
      rowi += 1
      if rowi >= len(less_data): rowi = 0
    return less_data
  
  diff = len(falling_data) - len(default_data)
  if diff > 0: default_data = augment(default_data, diff) # more falling, augment default
  else: falling_data = augment(falling_data, abs(diff)) # more default, augment falling
  return default_data, falling_data

def kerasQuantizeAndSave(model, inputs):
  # quantize
  def representative_dataset():
    for d in inputs: yield [tf.dtypes.cast(d, tf.float32)]

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8
  tflite_quant_model2 = converter.convert()
  with open('m2.tflite', 'wb') as f: f.write(tflite_quant_model2)

if __name__ == "__main__":
  model_export_dir, m1_name = "./tflite_models/", "m1.tflite"
  m1_data = "./datasets/m1_data" # subfolders are classification labels (still/moving), each contain frames resized to 224x224
  vids_dir = "./datasets/vids/" # subfolders are labels (default/falling), each contain videos resized to 224x224
  train_m2_with_custom_data, m2_custom_data_path = True, "./datasets/model2_data"
  

  '''
  generate model 1
  '''
  m1, loss, acc = trainM1(m1_data, 0.9)
  m1.export(export_dir = "./tflite_models/", tflite_filename = m1_name)

  '''
  generate data for training model 2
  '''
  if train_m2_with_custom_data: 
    default_data = np.loadtxt('./datasets/model2_data/default.txt')
    falling_data = np.loadtxt('./datasets/model2_data/falling.txt')
  else: # train on the dataset we just generated
    default_data = batchPredictMultVids(vids_dir + "default", model_export_dir + m1_name)
    falling_data = batchPredictMultVids(vids_dir + "falling", model_export_dir + m1_name)
  # augment data by gaussian error
  default_data, falling_data = augmentM2Data(default_data, falling_data)
  
  '''
  generate model 2
  '''
  m2 = trainM2(default_data, falling_data)
  kerasQuantizeAndSave(m2)
