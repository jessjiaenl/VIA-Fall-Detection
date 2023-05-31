import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Load data from text files
default_data = np.loadtxt('./../default.txt')
falling_data = np.loadtxt('./../falling.txt')

# Prepare input and output data
inputs = np.concatenate((default_data, falling_data))
outputs = np.concatenate((np.zeros(len(default_data)), np.ones(len(falling_data))))


# Convert inputs and outputs to TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

# Shuffle and batch the dataset
dataset = dataset.shuffle(len(inputs)).batch(128)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    dataset,
    epochs=6,
    
)

'''
# Train the model
model.fit(dataset, epochs=6)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1, 30)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)'''