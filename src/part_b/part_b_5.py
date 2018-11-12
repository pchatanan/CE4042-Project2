import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt
import time
from part_b_1 import read_data_chars
from part_b_2 import read_data_words


EMBED_SIZE = 20
HIDDEN_SIZE = 20
MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE_CHAR_1 = [20, 256]
FILTER_SHAPE_CHAR_2 = [20, 1]
FILTER_SHAPE_WORD_1 = [20, 20]
FILTER_SHAPE_WORD_2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

no_epochs = 100
lr = 0.01
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def char_cnn_model(x, keep):
    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE_CHAR_1,
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
        kernel_size=FILTER_SHAPE_CHAR_2,
        padding='VALID',
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME'
    )

    pool2 = tf.nn.dropout(tf.squeeze(tf.reduce_max(pool2, 1), axis=1), keep_prob=keep)

    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

    return input_layer, logits


def word_cnn_model(x, keep):
    word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=no_words, embed_dim=EMBED_SIZE)

    input_layer = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBED_SIZE, 1])

    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE_WORD_1,
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
        kernel_size=FILTER_SHAPE_WORD_2,
        padding='VALID',
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME'
    )

    pool2 = tf.nn.dropout(tf.squeeze(tf.reduce_max(pool2, 1), axis=1), keep)

    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

    return input_layer, logits


def char_rnn_model(x, keep):
    byte_vectors = tf.one_hot(x,256)
    byte_list = tf.unstack(byte_vectors, axis=1)

    cell_char = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, name='char')
    _, encoding = tf.nn.static_rnn(cell_char, byte_list, dtype=tf.float32)
    encoding = tf.nn.dropout(encoding, keep)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits


def word_rnn_model(x, keep):
    word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=no_words, embed_dim=EMBED_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)

    cell_word = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, name='word')
    _, encoding = tf.nn.static_rnn(cell_word, word_list, dtype=tf.float32)

    encoding = tf.nn.dropout(encoding, keep)
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits


def main():
    global no_words
    x_train_char, y_train_char , x_test_char, y_test_char = read_data_chars()
    x_train_word, y_train_word , x_test_word , y_test_word, no_words = read_data_words()


    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y = tf.placeholder(tf.int64)
    keep_prob = tf.placeholder_with_default(1.0, shape=())

    _, logits1 = char_cnn_model(x, keep_prob)
    _, logits2 = word_cnn_model(x, keep_prob)
    logits3 = char_rnn_model(x, keep_prob)
    logits4 = word_rnn_model(x, keep_prob)

    logits = [logits1, logits2, logits3, logits4]

    for i, l in enumerate(logits):
        entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, MAX_LABEL), logits=l))
        train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y, tf.argmax(l, axis=1)), tf.float32))

        if i<2:
            x_train, y_train, x_test, y_test = x_train_char, y_train_char , x_test_char, y_test_char
        else:
            x_train, y_train, x_test, y_test = x_train_word, y_train_word, x_test_word, y_test_word

        start_time = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss = []
            acc_test = []
            acc_train = []
            N = len(x_train)
            idx = np.arange(N)
            print('Current Model : {}'.format(i))
            print('iter: %d, entropy: %g, accuracy: %g   %g' % (0, entropy.eval(feed_dict={x: x_train, y: y_train}),
                                                                accuracy.eval(feed_dict={x: x_train, y: y_train}),
                                                                accuracy.eval(feed_dict={x: x_test, y: y_test})))
            for e in range(no_epochs):
                np.random.shuffle(idx)
                x_train, y_train = x_train[idx], y_train[idx]
                for (start, end) in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                    _, loss_, acc_ = sess.run([train_op, entropy, accuracy], {x: x_train[start:end], y: y_train[start:end], keep_prob: 0.5})
                loss.append(loss_)
                acc_train.append(acc_)
                acc_test.append(accuracy.eval(feed_dict={x:x_test, y:y_test}))

                if (e+1) % 5 == 0:
                    print('iter: %d, entropy: %g, accuracy: %g   %g' % (e+1, loss[e], acc_train[e], acc_test[e]))
        end_time = time.time()
        print('Time taken: %g' % (end_time-start_time))

        plt.figure()
        plt.plot(range(no_epochs), loss, 'b', label='Cross Entropy')
        plt.title('Training Cost against Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.legend(loc='lower right')
        plt.savefig('b5_{}_1.png'.format(i))

        plt.figure()
        plt.plot(range(no_epochs), acc_test, 'r', label='Accuracy')
        plt.title('Test Accuracy against Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig('b5_{}_2.png'.format(i))


if __name__ == '__main__':
    main()
