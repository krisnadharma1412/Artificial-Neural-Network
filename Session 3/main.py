# Back Propagation Neural Network (BPNN)
# -> Multilayer Perceptron (>2 layer)
# -> 3 Layer Type
#      a. Input Layer
#      b. Hidden Layer  
#      c. Output Layer

# Step 1 - Import Library
import tensorflow as tf # -> Process Data
import pandas as pd # -> Read File (.csv)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# Step 2 - Read File

def convert_to_integer(dataset_input):
    ordinal_encoder = OrdinalEncoder()
    return ordinal_encoder.fit_transform(dataset_input)

def normalize_data(dataset_input):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(dataset_input)

def make_class(dataset_output):
    one_hot_encoder = OneHotEncoder(sparse=False)
    return one_hot_encoder.fit_transform(dataset_output)



def import_data():
    data = pd.read_csv("LifeExpectancy.csv")
    
    dataset_input = data[["Gender", "Residential", "PhysicalActivity", "Happiness"]]
    dataset_output = data[["LifeExpectancy"]]
    
    dataset_input = convert_to_integer(dataset_input)
    dataset_input = normalize_data(dataset_input)

    dataset_output = make_class(dataset_output)

    return dataset_input, dataset_output

def feed_forward(dataset_input):
    process_input_hidden = tf.matmul(dataset_input, weight_input_hidden) + bias_input_hidden
    result_input_hidden = tf.nn.sigmoid(process_input_hidden)

    process_hidden_output = tf.matmul(result_input_hidden, weight_hidden_output) + bias_hidden_output
    result_hidden_output = tf.nn.sigmoid(process_hidden_output)

    return result_hidden_output

if  __name__ == '__main__':  
    dataset_input, dataset_output = import_data()
        
    train_dataset_input, test_dataset_input, train_dataset_output, test_dataset_output = \
        train_test_split(dataset_input, dataset_output, test_size=0.2) # train dataset 80% test dataset 20%

    layer = {
        "input" : 4,
        "hidden": 8,
        "output": 3
    }
    epoch = 1000
    learning_rate = 0.2 # 0 - 1

    # Placeholder
    placeholder_input = tf.placeholder(tf.float32, [None, layer["input"]]) # [?, 4]
    placeholder_output = tf.placeholder(tf.float32, [None, layer["output"]]) # [?, 3]

    # Weight
    weight_input_hidden = tf.Variable(tf.random_normal([layer["input"], layer["hidden"]]))
    weight_hidden_output = tf.Variable(tf.random_normal([layer["hidden"], layer["output"]]))

    # Bias
    bias_input_hidden = tf.Variable(tf.random_normal([layer["hidden"]]))
    bias_hidden_output = tf.Variable(tf.random_normal([layer["output"]]))

    prediction = feed_forward(placeholder_input)
    error = tf.reduce_mean(0.5 * (placeholder_output - prediction) ** 2) #0.5 * (pl_out - pre)^2
    update = tf.train.AdamOptimizer(learning_rate).minimize(error)
    # GradientDescentOptimizer

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            train_dictionary = {
                placeholder_input: train_dataset_input,
                placeholder_output: train_dataset_output
            }
            sess.run(update, feed_dict=train_dictionary)

            if i % 50 == 0:
                total_match = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(placeholder_output, axis=1))
                accuracy = tf.reduce_mean(tf.cast(total_match, tf.float32))

                test_dictionary = {
                    placeholder_input: test_dataset_input,
                    placeholder_output: test_dataset_output
                }
                print("Epoch: {} Accuracy: {}%".format(i, sess.run(accuracy, feed_dict=test_dictionary) * 100))

