import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt
import time


MAX_DOCUMENT_LENGTH = 100
EMBED_SIZE = 20
HIDDEN_SIZE = 20
MAX_LABEL = 15

no_epochs = 100
lr = 0.01
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def word_gru_model(x):
    word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=no_words, embed_dim=EMBED_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits


def word_vanilla_model(x):
    word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=no_words, embed_dim=EMBED_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits


def word_lstm_model(x):
    word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=no_words, embed_dim=EMBED_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding[1], MAX_LABEL, activation=None)

    return logits


def read_data_words():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('../../data/train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open('../../data/test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))
    y_train = y_train.values
    y_test = y_test.values

    return x_train, y_train, x_test, y_test, len(vocab_processor.vocabulary_)


def main():
    global no_words
    x_train, y_train, x_test, y_test, no_words = read_data_words()

    print(len(x_train))
    print(len(x_test))

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y = tf.placeholder(tf.int64)

    logits1 = word_gru_model(x)
    logits2 = word_vanilla_model(x)
    logits3 = word_lstm_model(x)
    logits = [logits1, logits2, logits3]
    loss_list = []
    acc_list = []

    for l in logits:
        # Optimizer
        entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, MAX_LABEL), logits=l))
        train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

        # accuracy
        num_correct = tf.cast(tf.equal(y, tf.argmax(l, axis=1)), tf.float32)
        accuracy = tf.reduce_mean(num_correct)

        start_time = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss = []
            acc_test = []
            N = len(x_train)
            idx = np.arange(N)
            loss.append(entropy.eval(feed_dict={x: x_train, y: y_train}))
            acc_test.append(accuracy.eval(feed_dict={x: x_test, y: y_test}))

            print('iter: %d, entropy: %g, accuracy:   %g' % (0, loss[0], acc_test[0]))
            for e in range(no_epochs):
                np.random.shuffle(idx)
                x_train, y_train = x_train[idx], y_train[idx]
                for (start, end) in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                    _, loss_ = sess.run([train_op, entropy], {x: x_train[start:end], y: y_train[start:end]})
                loss.append(loss_)
                acc_test.append(accuracy.eval(feed_dict={x:x_test, y:y_test}))

                if (e+1) % 5 == 0:
                    print('iter: %d, entropy: %g, accuracy: %g' % (e+1, loss[e], acc_test[e]))

        end_time = time.time()
        print('Time taken: %g' % (end_time - start_time))

        loss_list.append(loss)
        acc_list.append(acc_test)

    plt.figure()
    plt.plot(range(no_epochs+1), loss_list[0], 'b', label='GRU')
    plt.plot(range(no_epochs+1), loss_list[1], 'r', label='Vanilla')
    plt.plot(range(no_epochs+1), loss_list[2], 'g', label='LSTM')
    plt.title('Training Cost against Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend(loc='lower right')
    plt.savefig('b6_a_word_loss.png')

    plt.figure()
    plt.plot(range(no_epochs+1), acc_list[0], 'b', label='GRU')
    plt.plot(range(no_epochs+1), acc_list[1], 'r', label='Vanilla')
    plt.plot(range(no_epochs+1), acc_list[2], 'g', label='LSTM')
    plt.title('Test Accuracy against Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('b6_a_word_acc.png')


if __name__ == '__main__':
    main()
