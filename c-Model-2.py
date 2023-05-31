import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def normalize_sequence(sequence, label):
    """Normalizes sequence values: `uint8` -> `float32`."""
    return tf.cast(sequence, tf.float32) / 255., label

# Load data from text files
default_data = np.loadtxt('./datasets/model2_data/default.txt')
falling_data = np.loadtxt('./datasets/model2_data/falling.txt')

# Prepare input and output data
inputs = np.concatenate((default_data, falling_data))
outputs = np.concatenate((np.zeros(len(default_data)), np.ones(len(falling_data))))

# Split data into training and testing sets
train_size = int(0.8 * len(inputs))
train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
train_outputs, test_outputs = outputs[:train_size], outputs[train_size:]

# Convert inputs and outputs to TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_outputs))
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_outputs))

# Map the normalization function to the datasets
train_dataset = train_dataset.map(normalize_sequence, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(normalize_sequence, num_parallel_calls=tf.data.AUTOTUNE)

# Cache, shuffle, batch, and prefetch the datasets
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(len(train_inputs))
train_dataset = train_dataset.batch(128)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.cache()
test_dataset = test_dataset.batch(128)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=6, validation_data=test_dataset)

model.save("c_model2_export")

def normalize_sequence(sequence):
    """Normalizes sequence values: `uint8` -> `float32`."""
    return tf.cast(sequence, tf.float32) / 255.

#Test on a .txt file
def predict_falling(sequence, model):
    """Predicts if a sequence is falling using the trained model."""
    normalized_sequence = normalize_sequence(sequence)
    predictions = model.predict(np.expand_dims(normalized_sequence, axis=0))
    return predictions[0][0]

# Load the trained model
model = tf.keras.models.load_model('c_model2_export')
new_data = []
with open('./datasets/model2_data/falling.txt', 'r') as file:
    for line in file:
        sequence = [float(value) for value in line.strip().split()]
        new_data.extend(sequence)

new_data = np.array(new_data)

# Process the new data in sequences of 16 frames
sequence_length = 16
num_sequences = len(new_data) // sequence_length

for i in range(num_sequences):
    sequence = new_data[i * sequence_length: (i + 1) * sequence_length]
    is_falling = predict_falling(sequence, model)
    print(f"Sequence {i+1}: {'Falling' if is_falling > 0.5 else 'Not Falling'}")

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