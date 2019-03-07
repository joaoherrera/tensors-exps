# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# This exercise improves our very basic multilayer perceptron located on         #
# mperceptron.py by using tf.layers approach. These layers are an encapsulated   #
# version of what we did before, that is, create variables representing weights  #
# and perform some operations with them.                                         #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

def run(train_input_data, train_output_data, test_input_data, test_output_data, steps):
    # Let's start by creating or placeholders.
    # Now we gonna fill the first value of shape arguments with None.
    # This makes tensorflow handle with the current quantity of data.

    X = tf.placeholder(tf.float32, shape=[None, train_input_data.shape[1]], name='x')
    Y = tf.placeholder(tf.float32, shape=[None, train_output_data.shape[1]], name='y')

    # Instead of coding weights and its operations, let's create layers!
    first_layer = tf.layers.dense(inputs=X, units=train_input_data.shape[1])
    hidden_layer_1 = tf.layers.dense(inputs=first_layer, units=5)
    second_layer = tf.layers.dense(inputs=hidden_layer_1, units=train_output_data.shape[1])

    loss = tf.losses.sigmoid_cross_entropy(Y, second_layer)
    train = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(steps):
            sess.run(train, feed_dict={X: train_input_data, Y: train_output_data})
            netloss = sess.run(loss, feed_dict={X: train_input_data, Y: train_output_data})
            print('Loss: {0}'.format(netloss))

        # Calculate prediction accuracy
        predict = sess.run(second_layer, feed_dict={X: test_input_data})

    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) for estimate, target in zip(predict, test_output_data)]    
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
    run(xtrain, ytrain, xtest, ytest, 2000)
