# Import Library
import tensorflow as tf
import pandas as pd
import numpy as np 

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt 
import matplotlib.pyplot as plt

# Load and Normalize Dataset
def load_dataset():
    dataset = pd.read_csv("monthly-milk-production.csv",index_col = 'Month')
    dataset.index = pd.to_datetime(dataset.index)

    ratio = 0.9
    train_dataset = dataset[:int(len(dataset)*ratio)]
    test_dataset = dataset[int(len(dataset)*ratio)+1:]

    return train_dataset, test_dataset

train_dataset, test_dataset = load_dataset()


normalized_train_dataset = MinMaxScaler().fit_transform(train_dataset)
normalized_test_dataset = MinMaxScaler().fit_transform(train_dataset)


# Layer Architecture
layer = {
    'input': 1,
    'hidden': 20, #Bisa diganti
    'output': 1,
}

# Batch Preparation
batch_size = 2
time_step = 12

# Epoch and Learning Rate
epoch = 2000
learning_rate = 0.2

# Design Placeholder
feature_placeholder = tf.placeholder(tf.float32, [None, time_step, layer["input"]])
target_placeholder = tf.placeholder(tf.float32, [None, time_step, layer["output"]])

# Cell Construction + Activation Function
cell = tf.nn.rnn_cell.BasicRNNCell(layer["hidden"], activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=layer["output"], activation=tf.nn.relu)

# Dynamic RNN
output,_ = tf.nn.dynamic_rnn(cell, feature_placeholder, dtype=tf.float32)

# Calculate Loss
loss = tf.reduce_mean(0.5 * (target_placeholder-output) ** 2)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Tuning
saver = tf.train.Saver()

# Get Random Batch of Data
def get_batch(dataset, time_step, batch_size):
    input_batch = np.zeros(shape=(batch_size, time_step, layer["input"]))
    output_batch = np.zeros(shape=(batch_size, time_step, layer["output"]))

    for i in range(batch_size):
        point = np.random.randint(0,len(dataset)-time_step)
        
        input_batch[i] = dataset[point:point+time_step]
        output_batch[i] = dataset[point+1:point+time_step+1]

    return input_batch, output_batch

# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch+1):
        input_batch, output_batch = get_batch(normalized_train_dataset, time_step, batch_size)

        feed = {
            feature_placeholder: input_batch,
            target_placeholder: output_batch
        }

        sess.run(train, feed_dict=feed)
        if(i%250 == 0):
            print("Epoch: {}, Loss: {}".format(i, sess.run(loss, feed_dict=feed)))

    saver.save(sess, "model/model.ckpt")

# Test
with tf.Session() as sess:
    saver.restore(sess, "model/model.ckpt")

    listed_dataset = list(normalized_train_dataset)
    input_batch = np.array(listed_dataset[-time_step -1:len(listed_dataset)-1]).reshape(1, time_step, layer["input"])

    feed = {
        feature_placeholder: input_batch
    }

    prediction = sess.run(output, feed_dict=feed)
    for i in range(time_step):
        listed_dataset.append(prediction[0,i,0])

prediction_calculation = np.array(listed_dataset[-time_step]).reshape([1, time_step])
prediction_calculation =MinMaxScaler().fit(normalized_test_dataset).inverse_transform(prediction_calculation)

visualizer = train_dataset.tail(time_step)
visualizer["prediction"] = prediction_calculation[0]

visualizer.plot()
plt.show 