import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load Dataset
def load_dataset():
    dataset = pd.read_csv('lie-dataset.csv')

# Normalize Dataset
    features =dataset[['HeartRate', 'BlinkFreq', 'EyeContact']]
    targets = dataset[['Class']]

    features = MinMaxScaler().fit_transform(features)
    targets = OneHotEncoder(sparse=False).fit_transform(targets)


    return train_test_split(features, targets, test_size=0.2)

input_train, input_test, output_train, output_test = load_dataset()

# Architecture
layer ={
    'input': 3,
    'hiddenone': 6,
    'hiddentwo': 3,
    'output': 2,
}

# Stochastic Weight and Bias
weight = {
    'input_to_hiddenone': tf.Variable(tf.random.normal([layer['input'], layer['hiddenone']])), 
    'hiddenone_to_hiddentwo': tf.Variable(tf.random.normal([layer['hiddenone'], layer['hiddentwo']])),
    'hiddentwo_to_output': tf.Variable(tf.random.normal([layer['hiddentwo'], layer['output']])) 
}

bias = {
    'input_to_hiddenone': tf.Variable(tf.random.normal([layer['hiddenone']])), 
    'hiddenone_to_hiddentwo': tf.Variable(tf.random.normal([layer['hiddentwo']])),
    'hiddentwo_to_output': tf.Variable(tf.random.normal([layer['output']])) 
}

features_placeholder = tf.compat.v1.placeholder(tf.float32, [None, layer['input']])
targets_placeholder = tf.compat.v1.placeholder(tf.float32, [None, layer['output']])

# Feed Forward
def feed_forward():
    weightxbias1 = tf.matmul(features_placeholder, weight['input_to_hiddenone'])+bias['input_to_hiddenone']
    activation1 = tf.nn.sigmoid(weightxbias1)

    weightxbias2 = tf.matmul(activation1, weight['hiddenone_to_hiddentwo'])+bias['hiddenone_to_hiddentwo']
    activation2 = tf.nn.sigmoid(weightxbias2)

    weightxbias3 = tf.matmul(activation2, weight['hiddentwo_to_output'])+bias['hiddentwo_to_output']
    activation3 = tf.nn.sigmoid(weightxbias3)

    return activation3

# Define Epoch and Learning Rate
epoch = 5000
learning_rate = 0.1

# Calculate Loss + Optimize
output = feed_forward()
loss = tf.reduce_mean(0.5 *(output - targets_placeholder) ** 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {
        features_placeholder: input_train,
        targets_placeholder: output_train
    }

    for i in range(1, epoch+1):
        sess.run(train, feed_dict=feed)

        if(i % 200 == 0):
            print("Epoch: %d, Loss: %.2f%%"%(i, sess.run(loss, feed_dict=feed)))

    # Test

    feed = {
        features_placeholder: input_test,
        targets_placeholder: output_test
    }
    match = tf.equal(tf.argmax(output,axis=1),tf.argmax(targets_placeholder,axis=1))
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))

    print("Accuracy: {}%".format(sess.run(accuracy, feed_dict=feed)*100))
