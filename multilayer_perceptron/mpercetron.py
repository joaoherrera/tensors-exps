# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# This exercise was based on articles:                                           #
# https://stackabuse.com/tensorflow-neural-network-tutorial/                     #
# https://jacobbuckman.com/post/tensorflow-the-confusing-parts-1/                #
#                                                                                #
#                Multilayer perceptron using Tensorflow library                  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


def train(input_data, labels, neurons_hidden_layer, iterations):

    # Take note!
    # Tensorflow has three different types of tensors:
    # . Constants: values that doesn't change on run time
    # . Variables: values that may change during run time. Ie. neurons weights, biases, etc. 
    # . Placeholders: values from external sources. They represent a "promise" that a value will be
    # provided when the graph is run. Ie. Input data, labels, etc.

    tf.reset_default_graph()

    # So now, let's create a placeholder to store the input and other to store the output values.
    X = tf.placeholder(dtype=tf.float32, shape=input_data.shape, name='x')
    Y = tf.placeholder(dtype=tf.float32, shape=labels.shape, name='y')

    # Variables for the two groups of weights among the three layers of the network
    w1 = tf.Variable(np.random.rand(input_data.shape[1], neurons_hidden_layer), dtype=tf.float32)
    w2 = tf.Variable(np.random.rand(neurons_hidden_layer, labels.shape[1]), dtype=tf.float32)

    out1 = tf.sigmoid(tf.matmul(X, w1))  # Multiply every value of input data by weights in w1
    out2 = tf.sigmoid(tf.matmul(out1, w2))  # Same to hidden to output layer

    # define a loss function
    deltas = tf.square(out2 - Y)
    loss = tf.reduce_sum(deltas)

    # define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for _ in range(iterations):
        sess.run(train, feed_dict={X: input_data, Y:labels})

        netloss = sess.run(loss, feed_dict={X: input_data, Y: labels})
        print('Loss: {0}'.format(netloss))

        weights_1 = sess.run(w1)
        weights_2 = sess.run(w2)

    return weights_1, weights_2


def test(input_data, labels, weights_1, weights_2):

    # Create placeholders to store input and output data
    X = tf.placeholder(tf.float32, shape=input_data.shape, name='x')
    Y = tf.placeholder(tf.float32, shape=labels.shape, name='y')

    # weights are now constants
    w1 = tf.constant(weights_1, tf.float32)
    w2 = tf.constant(weights_2, tf.float32)

    out1 = tf.sigmoid(tf.matmul(X, w1))
    out2 = tf.sigmoid(tf.matmul(out1, w2))

    with tf.Session() as sess:
        predict = sess.run(out2, feed_dict={X: input_data, Y: labels})

    # Calculate prediction accuracy
    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) for estimate, target in zip(predict, labels)]
    
    print('Accuracy: {0}'.format(np.sum(correct) / len(correct)))


if __name__ == '__main__':
    # load iris dataset from sklearn.
    # we can also download the .csv file from tensorflow repo.
    dataset = load_iris()

    Xdata = dataset['data']
    Ydata = label_binarize(dataset['target'], [0, 1, 2])
    neurons_hidden_layer = 10

    # split iris data into train and test subsets.
    xtrain, xtest, ytrain, ytest = train_test_split(Xdata, Ydata, train_size=0.8, test_size=0.2)

    weights_1, weights_2 = train(xtrain, ytrain, neurons_hidden_layer, 2000)
    test(xtest, ytest, weights_1, weights_2)
