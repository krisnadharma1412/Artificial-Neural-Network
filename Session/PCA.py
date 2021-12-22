import tensorflow as tf
from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt

# Covariance
# Cov = (x1-xmean1)(x2-xmean2)Transpose
# EigenValue
# Val = data - EigenVector

def load_dataset():
    dataset = loadmat('olivettifaces.mat')
    print(dataset)
    dataset = dataset['faces']
    transposed_dataset = np.transpose(dataset)
    return dataset, transposed_dataset.astype(float)
    

def calculate_mean(dataset):
    return tf.reduce_mean(dataset, axis=0)

def normalize(dataset, mean):
    return dataset - mean

def calculate_covariance(eigen_value):
    return tf.matmul(eigen_value, tf.transpose(eigen_value))

def calculate_eigen_vector(covariance_value):
    eigen_value, eigen_vector = tf.self_adjoint_eig(covariance_value)
    return tf.reverse(eigen_vector, [1])

def calculate_eigen_face(dataset, eigen_vector):
    return tf.transpose(tf.matmul(tf.transpose(dataset)
    , eigen_vector))

def plot_image(image):
    image = image.reshape(64,64)
    image = np.transpose(image)
    plt.imshow(image)
    plt.show()

original_dataset, transposed_dataset = load_dataset()
xmean = calculate_mean(transposed_dataset)
normalized_dataset = normalize(transposed_dataset,
xmean)
covariance_value = calculate_covariance(normalized_dataset)
eigen_vector = calculate_eigen_vector(covariance_value)
eigen_faces = calculate_eigen_face(normalized_dataset, eigen_vector)

with tf.Session() as sess:
    result = sess.run(eigen_faces)

plot_image(result[5])

#Cara ke 2 JAUUH lebih singkat
from sklearn.decomposition import pca

dataset = loadmat('olivettifaces.mat')
pca = PCA(n_components=4)
dataset = pca.fit_transform(dataset)
