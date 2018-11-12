import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt
import time


MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

no_epochs = 100
lr = 0.01
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def char_cnn_model(x):
    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME'
    )

    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), axis=1)

    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

    return input_layer, logits


def read_data_chars():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('../../data/train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[1])
            y_train.append(int(row[0]))

    with open('../../data/test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[1])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)

    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(char_processor.fit_transform(x_train)))
    x_test = np.array(list(char_processor.transform(x_test)))
    y_train = y_train.values
    y_test = y_test.values

    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = read_data_chars()

    print(len(x_train))
    print(len(x_test))

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y = tf.placeholder(tf.int64)

    inputs, logits = char_cnn_model(x)

    # Optimizer
    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    # accuracy
    num_correct = tf.cast(tf.equal(y, tf.argmax(logits, axis=1)), tf.float32)
    accuracy = tf.reduce_mean(num_correct)

    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss = []
        acc_test = []
        acc_train = []
        N = len(x_train)
        idx = np.arange(N)
        print('iter: %d, entropy: %g, accuracy: %g   %g' % (0, entropy.eval(feed_dict={x: x_train, y: y_train}),
                                                            accuracy.eval(feed_dict={x: x_train, y: y_train}),
                                                            accuracy.eval(feed_dict={x: x_test, y: y_test})))
        for e in range(no_epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]
            for (start, end) in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                _, loss_, acc_ = sess.run([train_op, entropy, accuracy], {x: x_train[start:end], y: y_train[start:end]})
            loss.append(loss_)
            acc_train.append(acc_)
            acc_test.append(accuracy.eval(feed_dict={x:x_test, y:y_test}))

            if e % 5 == 0 or e == no_epochs-1:
                print('iter: %d, entropy: %g, accuracy: %g   %g' % (e+1, loss[e], acc_train[e], acc_test[e]))
    end_time = time.time()
    print('Time taken: %g' % (end_time-start_time))

    plt.figure(1)
    plt.plot(range(no_epochs), loss, 'b', label='Cross Entropy')
    plt.title('Training Cost against Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend(loc='lower right')
    plt.savefig('b1_1.png')

    plt.figure(2)
    plt.plot(range(no_epochs), acc_test, 'r', label='Accuracy')
    plt.title('Test Accuracy against Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('b1_2.png')


if __name__ == '__main__':
    main()
