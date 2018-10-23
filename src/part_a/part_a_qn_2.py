#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import random
import pylab as plt
import pickle

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 60
batch_size = 128
NUM_FEATURE_MAPS = [40, 80, 120, 160, 200, 240]

seed = 10
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels - 1] = 1

    return data, labels_


def cnn(images, c1, c2):
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # First convolutional layer - maps one RGB image to c1 feature maps.
    W_conv1 = weight_variable([9, 9, NUM_CHANNELS, c1], 1.0 / np.sqrt(NUM_CHANNELS * 9 * 9), 'weights_1')
    b_conv1 = bias_variable([c1], 'biases_1')
    u_conv1 = tf.nn.conv2d(images, W_conv1, [1, 1, 1, 1], padding='VALID') + b_conv1
    h_conv1 = tf.nn.relu(u_conv1)

    # First Pooling layer - downsamples by 2X.
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h_pool1')

    # Second convolutional layer - maps c1 feature maps to c2.
    W_conv2 = weight_variable([5, 5, c1, c2], 1.0 / np.sqrt(c1 * 5 * 5), 'weights_2')
    b_conv2 = bias_variable([c2], 'biases_2')
    u_conv2 = tf.nn.conv2d(h_pool1, W_conv2, [1, 1, 1, 1], padding='VALID') + b_conv2
    h_conv2 = tf.nn.relu(u_conv2)

    # Second Pooling layer - downsamples by 2X.
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h_pool2')

    # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
    # is down to 8x8x60 feature maps -- maps this to 300 features.

    dim = h_pool2.get_shape()[1].value * h_pool2.get_shape()[2].value * h_pool2.get_shape()[3].value
    h_pool2_flat = tf.reshape(h_pool2, [-1, dim])

    W_fc1 = weight_variable([dim, 300], 1.0 / np.sqrt(dim), 'weights_fc1')
    b_fc1 = bias_variable([300], 'biases_fc1')

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Softmax layer
    W_fc2 = weight_variable([300, 10], 1.0 / np.sqrt(300), 'weights_fc2')
    b_fc2 = bias_variable([10], 'biases_fc2')

    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    return logits


def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    return tf.Variable(tf.zeros(shape), name=name)


def main():
    trainX, trainY = load_data('../../data/data_batch_1')
    x_min, x_max = np.min(trainX, axis=0), np.max(trainX, axis=0)
    print(trainX.shape, trainY.shape)

    testX, testY = load_data('../../data/test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - x_min) / (x_max - x_min)
    testX = (testX - x_min) / (x_max - x_min)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    all_test_acc = {}
    last_test_acc = {}
    for c1 in range(len(NUM_FEATURE_MAPS)):
        for c2 in range(c1 + 1, len(NUM_FEATURE_MAPS)):
            num_c1, num_c2 = NUM_FEATURE_MAPS[c1], NUM_FEATURE_MAPS[c2]
            label = "{}-->{}".format(num_c1, num_c2)
            print("Feature maps combination: {}".format(label))

            logits = cnn(x, num_c1, num_c2)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
            loss = tf.reduce_mean(cross_entropy)

            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

            N = len(trainX)
            idx = np.arange(N)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                test_acc = []
                for e in range(epochs):
                    np.random.shuffle(idx)
                    trainX, trainY = trainX[idx], trainY[idx]

                    for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                        _ = sess.run([train_step], {x: trainX[start:end], y_: trainY[start:end]})

                    accuracy_ = sess.run([accuracy], {x: testX, y_: testY})
                    test_acc.append(accuracy_[0])

                    if e % (epochs // 5) == 0 or e == epochs - 1:
                        print('epoch {0:5d}: Test Acc: {1:8.4f}'.format(e + 1, test_acc[e]))

                all_test_acc[label] = test_acc
                last_test_acc[label] = test_acc[-1]

    sorted_last_test_acc = {}
    for key in sorted(last_test_acc, key=last_test_acc.get, reverse=True)[:5]:
        sorted_last_test_acc.update({key: last_test_acc[key]})

    print(sorted_last_test_acc)

    # plot learning curves
    plt.figure(1)
    for key_label in sorted_last_test_acc:
        plt.plot(range(epochs), all_test_acc.get(key_label), label=key_label)
    plt.title('Test Accuracy against Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend(loc='best')
    plt.savefig('./Qn2/part_a_qn2.png')

    plt.show()


if __name__ == '__main__':
    main()
