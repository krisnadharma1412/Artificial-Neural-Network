# import library
from io import BufferedIOBase
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder

# Load dataset 'mnist'

def plot_image(input_dataset):
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(input_dataset[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()

(training_input, training_output), (testing_input, testing_output) = mnist.load_data()
# plot_image(training_input)
# plot_image(testing_input)

# print(training_input)
# print(testing_input)
# print(training_output)
# print(testing_output)

# Convert output dataset from [x, x, x, ..] to [[x], [x], [x], ...]
training_output = training_output.reshape(-1, 1)
testing_output = testing_output.reshape(-1, 1)
# print(training_output)
# print(testing_output)

# Normalize dataset (One Hot Encoder)
# sparse=True return matrix, sparse=false return array
encoder = OneHotEncoder(sparse=False).fit(training_output)
training_output = encoder.transform(training_output)
testing_output = encoder.transform(testing_output)
# print(training_output)
# print(testing_output)

# Initialize image size (width and height)
width = 28
height = 28

# Initialize placeholder (input and output)
placeholder_input = tf.placeholder(tf.float32, [None, width, height])
placeholder_output = tf.placeholder(tf.float32, [None, training_output.shape[1]])
placeholder_input_image = tf.reshape(placeholder_input, [-1, width, height, 1])

# Initialize convolutional layer (weight and bias)
def create_convolutional_layer(shape):
    weight = tf.Variable(tf.random_normal(shape))
    bias = tf.Variable(tf.random_normal([shape[3]]))
    return{
        'weight': weight,
        'biase': bias
    }
conv1_layer = create_convolutional_layer([5, 5, 1, 8])
conv2_layer = create_convolutional_layer([5, 5, 8, 16])
# print(conv1_layer)
# print(conv2_layer)

# Calculate convolutional layer and pooling layer (layer 1 and layer 2)
def convolve(x, layer):
    y = tf.nn.conv2d(x, strides=1, padding='SAME') + LAYER['bias']
    return tf.nn.relu(y)

def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv1 = convolve(placeholder_input_image, conv1_layer)
conv1_pooled = pool(conv1)
conv2 = convolve(conv1_pooled, conv2_layer)
conv2_pooled = pool(conv2)
# print(conv1)
# print(conv1_pooled)
# print(conv2)
# print(conv2_pooled)

# Reshape result of convolutional & pooling from (?, 7, 7, 16) to (?, 784)
flat_input = tf.reshape(conv2__pooled, [-1, 7 * 7 * 16])
# print(flat_input)

# Create fully connected layers
def fully_connected(x, output_size):
    input_size = int(x.shape[1])
    weight = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.random_normal([output_size]))
    return tf.matmul(x, weight) + bias
prediction_output = fully_connected(flat_input, training_output.shape[1])

# Create error/loss function
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=placeholder_output,
    logits=prediction_output
))

# Update weight
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(error)

# Create batch method
def get_batch(iteration, x, y, batch_size):
    start_index = iteration * batch_size
    total_data = len(x)

    if total_data <= start_index:
        return None
    x_batch = x[start_index : min(start_index + batch_size, total_data)]
    y_batch = y[start_index : min(start_index + batch_size, total_data)]
    return x_batch, y_batch

epoch = 10
batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variable_initializer())

    for i in range(1, epoch+1):
        iteration = 0
        batch_data = get_batch(iteration, training_input, training_output, batch_size)

        while batch_data is not None:
            x_batch, y_batch = batch_data
            sess.run(train, feed_dict={
                placeholder_input: x_batch,
                placeholder_output: y_batch
            })

            iteration += 1

            batch_data = get_batch(iteration, training_input, training_output, batch_size)

            if iteration % 200 == 0:
                matches = tf.equal(tf.argmax(placeholder_output, axis=1), 
                          tf.argmax(prediction_output), axis=1)
                accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
                print(f'Epoch: {i} | 
                        Iteration: {iteration} | 
                        Accuracy: {sess.run(accuracy, feed_dict={placeholder_input: testing_input, placeholder_output: testing_output})}')


