# Import Library

import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
import numpy as np 

# Preprocessing dataset
dataset = pd.read_csv("Milk.csv", index_col="Month")

# Convert Index dtype from ""
dataset.index = pd.to_datetime(dataset.index)

# Visualize dataset
# plt.title("Milk Production")
# plt.plot(dataset)
# plt.show()

# Split Dataset
testing_size = int(len(dataset) / 8)
training = dataset.head(len(dataset) - testing_size)
testing = dataset.tail(testing_size)

# Normalize Dataset
scaler = MinMaxScaler()
training_dataset = scaler.fit_transform(training)
testing_dataset = scaler.fit_transform(testing)

# Initialize Variable
input_layer = 1
output_layer = 1
context_layer = 80
time_steps = 20
batch_size = 2      # Number of samples processed before the model updated
epoch = 2000        # Number of complete training dataset

learning_rate = 0.02


# Create RNN Architecture
# GRUCell -> Gate Recurrent Unit
# BasicLSTMCell -> Long Short Term Memory
# LSTMCell
# BasicRNNCell
# MultiRNNCell
# DropoutWrapper

# Create Basic
cell = tf.nn.rnn_cell.BasicRNNCell(context_layer, activation=tf.nn.relu)

# Connect Context Layer to Output Layer
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=output_layer, activation=tf.nn.relu)

# Create Placeholder
x = tf.placeholder(tf.float32, [None, time_steps, input_layer])
y = tf.placeholder(tf.float32, [None, time_steps, output_layer])

# Connect Input Placeholder to RNN Architecture
output, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

# Create Loss and Optimizer

# Create Loss/Error (MSE - Mean Squared Error)
# 1/2 * (output-prediction)^2
error = tf.reduce_mean(0.5 * (y - output) ** 2)

# Create Optimizer
train = tf.train.AdamOptimizer(learning_rate).minimize(error)


# Create Training Batch Method

def batch(dataset, time_steps, batch_size):
    
    # np.zeros -> Set Value to 0
    x_batch = np.zeros(shape=(batch_size, time_steps, input_layer))
    y_batch = np.zeros(shape=(batch_size, time_steps, output_layer))

    # [start:end]
    for i in range(batch_size):
        random = np.random.randint(0, len(dataset) - time_steps)
        x_batch[i] = dataset[random: random + time_steps]
        y_batch[i] = dataset[random + 1: random + time_steps + 1]

    return x_batch, y_batch

# Training

# Create Object to Save Training Data / Model
saver = tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        # Train Dataset
        x_batch, y_batch = batch(dataset, time_steps, batch_size)
        sess.run(train, feed_dict={
            x: x_batch,
            y: y_batch
        })

        # Print Loss/Error Every 100 Epoch
        if i % 100 == 0:
            print("Iteration: {}, Loss: {}".format(i, sess.run(error, feed_dict={
                x: x_batch,
                y: y_batch
            })))
    saver.save(sess, 'models/milk.ckpt')

# Testing

with tf.Session() as sess:

    # Restore Training Model
    saver.restore(sess, 'models/milk.ckpt')

    # Convert Dataset to List
    testing_dataset_list = list(testing_dataset)

    # Testing Dataset
    for i in range(testing_size):
        x_batch = np.array(testing_dataset_list[-time_steps:]).reshape(1, time_steps, input_layer)
        prediction = sess.run(output, feed_dict={
            x_batch
        })
        testing_dataset_list.append(prediction[0, -1, 0])

# Re-Transform Testing Dataset
prediction_result = scaler.inverse_transform(
    np.array(testing_dataset_list[-testing_size:])).reshape(testing_size, 1)

compare = testing.tail(testing_size)
compare['Prediction'] = prediction_result

plt.title("Milk Production")
plt.plot(compare)
plt.show()

# Gradient Problem
# Vanishing Gradient -> Prediction always reached minimum
# Exploding Gradient -> Prediction always reached maximum