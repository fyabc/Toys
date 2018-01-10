#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

__author__ = 'fyabc'

DataPath = 'G:/Data/PatternRecognition/preprocessing/output_normalized'
RealDataPath = 'G:/Data/PatternRecognition/preprocessing/test_hit'


def load_data(merge=True):
    all_data = {'hit': [], 'be_hit': [], 'others': []}
    for sensor in os.listdir(DataPath):
        for t in ['hit', 'be_hit', 'others']:
            for fn in os.listdir(os.path.join(DataPath, sensor, t)):
                data = np.load(os.path.join(DataPath, sensor, t, fn))
                if merge:
                    for image in data:
                        all_data[t].append(image[:-1])
                else:
                    all_data[t].append(data[:, :-1])

    return all_data


def load_real_data(filenames=('1.npy', '2.npy')):
    result = [np.load(os.path.join(RealDataPath, filename))[:, :-1] for filename in filenames]
    return np.vstack(result)


# Plot utils.
def average_without_none(l):
    no_none = [e for e in l if e is not None]

    if not no_none:
        return None

    return sum(no_none) / len(no_none)


def move_avg(l, mv_avg=5):
    return [average_without_none(l[max(i - mv_avg, 0):i + 1]) for i in range(len(l))]


class Config:
    """Configuration class."""

    prob_size = 3
    input_size = 34

    batch_size = 20
    hidden_size = 64
    iteration = 150

    num_steps = 100
    keep_prob = 1.
    n_layers = 1
    max_grad_norm = 5
    optimizer = 'adam'
    learning_rate = 0.001
    max_epoch = 20
    init_scale = 0.1
    l2_norm = 0.0005

    use_mask = True
    tag_label = False
    tag_hidden_size = 64

    # Plot options.
    real_move_avg = 5
    merge_real_into_test = True


class MLPConfig(Config):
    hidden_size = 24
    iteration = 1500
    real_move_avg = 1


def build_model(x, config: Config):
    W1 = tf.Variable(tf.truncated_normal([config.input_size, config.hidden_size], stddev=0.1), name='W1')
    b1 = tf.Variable(tf.constant(0.1, shape=[config.hidden_size]), name='b1')
    h = tf.matmul(x, W1) + b1
    W2 = tf.Variable(tf.truncated_normal([config.hidden_size, config.prob_size], stddev=0.1), name='W2')
    b2 = tf.Variable(tf.constant(0.1, shape=[config.prob_size]), name='b2')
    y = tf.matmul(h, W2) + b2

    return y


def build_rnn_model(x, is_training, config: Config):
    """x: ([B], [T], [W])"""

    ff_in_w = tf.get_variable('ff_in_w', [config.input_size, config.hidden_size], dtype=tf.float32)
    ff_in_b = tf.get_variable('ff_in_b', [config.hidden_size], dtype=tf.float32)
    x = tf.matmul(tf.reshape(x, [-1, config.input_size]), ff_in_w) + ff_in_b
    x = tf.reshape(x, [-1, config.num_steps, config.hidden_size])

    def _lstm_cell():
        return rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)

    def _attn_cell():
        if is_training and config.keep_prob < 1:
            return rnn.DropoutWrapper(_lstm_cell(), output_keep_prob=config.keep_prob)
        else:
            return _lstm_cell()

    cell = rnn.MultiRNNCell([_attn_cell() for _ in range(config.n_layers)], state_is_tuple=True)

    _init_state = cell.zero_state(config.batch_size, tf.float32)

    # Outputs: ([B], [T], [H])
    with tf.variable_scope('RNN') as vs:
        outputs, _final_state = tf.nn.dynamic_rnn(
            cell, x,
            initial_state=_init_state,
            time_major=False,
            scope=vs,
        )

    # Output: ([B], [T] * [H])
    output = tf.reshape(outputs, [outputs.shape[0], -1])

    # Feed-forward.
    ff_w = tf.get_variable('ff_w', [config.num_steps * config.hidden_size, config.prob_size], dtype=tf.float32)
    ff_b = tf.get_variable('ff_b', [config.prob_size], dtype=tf.float32)

    logits = tf.matmul(output, ff_w) + ff_b

    return logits


def main():
    config = MLPConfig()
    all_data = load_data()

    real_data = load_real_data()
    real_labels = np.full(real_data.shape[:1], 1, dtype='int64')

    data = all_data['hit'] + all_data['be_hit'] + all_data['others'][:4500]
    labels = np.array([1] * len(all_data['hit']) + [2] * len(all_data['be_hit']) + [0] * 4500)
    data = np.array(data)

    index = np.arange(len(data), dtype='int64')
    np.random.shuffle(index)
    data = data[index]
    labels = labels[index]

    test_data = data[:1000]
    test_labels = labels[:1000]
    data = data[1000:]
    labels = labels[1000:]
    dev_data = data[:1000]
    dev_labels = labels[:1000]

    if config.merge_real_into_test:
        test_data = np.concatenate([test_data, real_data])
        test_labels = np.concatenate([test_labels, real_labels])

    x = tf.placeholder(tf.float32, shape=[None, config.input_size], name='x')
    y = tf.placeholder(tf.int64, shape=[None], name='y')
    y_predicted = build_model(x, config)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_predicted))
    # loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * config.l2_norm
    train_op = tf.train.AdamOptimizer().minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y_predicted, 1), y), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_list = []
        train_acc_list = []
        dev_acc_list = []
        test_acc_list = []
        real_acc_list = []

        for i in range(config.iteration):
            _slice = slice(i * config.batch_size % len(labels), (i + 1) * config.batch_size % len(labels))
            if _slice.start >= _slice.stop:
                continue
            _, loss_val, accuracy_val = sess.run([train_op, loss, accuracy], feed_dict={
                x: data[_slice],
                y: labels[_slice],
            })

            dev_accuracy = sess.run(accuracy, feed_dict={
                x: dev_data,
                y: dev_labels,
            })

            test_accuracy = sess.run(accuracy, feed_dict={
                x: test_data,
                y: test_labels,
            })

            real_accuracy = sess.run(accuracy, feed_dict={
                x: real_data,
                y: real_labels,
            })

            print('Iter: {}, loss: {:.6f}, batch acc: {:.6f}, dev acc: {:.6f}, '
                  'test acc: {:.6f}, real acc: {:.6f}'.format(
                    i, loss_val, accuracy_val, dev_accuracy, test_accuracy, real_accuracy))
            loss_list.append(loss_val)
            train_acc_list.append(accuracy_val)
            dev_acc_list.append(dev_accuracy)
            test_acc_list.append(test_accuracy)
            real_acc_list.append(real_accuracy)

    plt.subplot(121)
    plt.plot(loss_list, label=r'$Loss$')
    plt.legend(loc='best')
    plt.xlabel(r'$Iteration$')
    plt.ylabel(r'$Loss$')
    plt.grid()

    plt.subplot(122)
    plt.plot(dev_acc_list, label='验证集')
    plt.plot(test_acc_list, label='测试集')
    if not config.merge_real_into_test:
        plt.plot(move_avg(real_acc_list, config.real_move_avg), label='网络来源测试集')
    plt.legend(loc='best')
    plt.grid(which='both')
    plt.ylim(ymin=0.4, ymax=1.0)
    plt.xlabel(r'$Iteration$')
    plt.ylabel(r'$Accuracy$')

    plt.suptitle('MLP')

    plt.gcf().set_size_inches(12.80, 4.80)
    plt.savefig('PRLabCurves/result_mlp.png')
    # plt.show()
    plt.clf()


def main_rnn():
    all_data = load_data(merge=False)
    real_data = load_real_data()

    config = Config()

    all_real_data = []
    for _ in range(config.batch_size * 10):
        start = np.random.randint(0, len(real_data) - config.num_steps)
        all_real_data.append(real_data[start: start + config.num_steps])
        assert len(all_real_data[-1]) == config.num_steps
    real_data = np.array(all_real_data)
    real_labels = np.full(real_data.shape[:1], 1, dtype='int64')

    # data: ([N], [T], [W])
    all_hit = []
    for image in all_data['hit']:
        for _ in range(len(image) // config.num_steps * 10):
            start = np.random.randint(0, len(image) - config.num_steps)
            all_hit.append(image[start: start + config.num_steps])
            assert len(all_hit[-1]) == config.num_steps

    all_be_hit = []
    for image in all_data['be_hit']:
        for _ in range(len(image) // config.num_steps * 10):
            start = np.random.randint(0, len(image) - config.num_steps)
            all_be_hit.append(image[start: start + config.num_steps])
            assert len(all_be_hit[-1]) == config.num_steps

    all_other = []
    _cnt = 0
    for image in all_data['others']:
        if _cnt > len(all_hit):
            break
        for _ in range(len(image) // config.num_steps * 10):
            start = np.random.randint(0, len(image) - config.num_steps)
            all_other.append(image[start: start + config.num_steps])
            _cnt += 1
            if _cnt > len(all_hit):
                break
            assert len(all_other[-1]) == config.num_steps

    labels = np.array([1] * len(all_hit) + [2] * len(all_be_hit) + [0] * len(all_other))
    data = np.array(all_hit + all_be_hit + all_other)

    index = np.arange(len(data), dtype='int64')
    np.random.shuffle(index)
    data = data[index]
    labels = labels[index]

    test_data = data[:200]
    test_labels = labels[:200]
    data = data[200:]
    labels = labels[200:]
    dev_data = data[:200]
    dev_labels = labels[:200]

    if config.merge_real_into_test:
        test_data = np.concatenate([test_data, real_data])
        test_labels = np.concatenate([test_labels, real_labels])

    x = tf.placeholder(tf.float32, shape=[None, config.num_steps, config.input_size], name='x')
    y = tf.placeholder(tf.int64, shape=[None], name='y')
    y_predicted = build_rnn_model(x, is_training=True, config=config)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_predicted))
    loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * config.l2_norm
    train_op = tf.train.AdamOptimizer().minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y_predicted, 1), y), tf.float32))

    def _run_acc(session, config, acc_fn, data, labels):
        all_acc = []
        for j in range(len(labels) // config.batch_size):
            all_acc.append(session.run(acc_fn, feed_dict={
                x: data[j * config.batch_size: (j + 1) * config.batch_size],
                y: labels[j * config.batch_size: (j + 1) * config.batch_size],
            }))
        return np.mean(all_acc)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_list = []
        train_acc_list = []
        dev_acc_list = []
        test_acc_list = []
        real_acc_list = []

        for i in range(config.iteration):
            _slice = slice(i * config.batch_size % len(labels), (i + 1) * config.batch_size % len(labels))
            data_slice, labels_slice = data[_slice], labels[_slice]
            if len(labels_slice) < config.batch_size:
                continue
            _, loss_val, accuracy_val = sess.run([train_op, loss, accuracy], feed_dict={
                x: data[_slice],
                y: labels[_slice],
            })

            dev_accuracy = _run_acc(sess, config, accuracy, dev_data, dev_labels)
            test_accuracy = _run_acc(sess, config, accuracy, test_data, test_labels)
            real_accuracy = _run_acc(sess, config, accuracy, real_data, real_labels)

            print('Iter: {}, loss: {:.6f}, batch acc: {:.6f}, dev acc: {:.6f}, '
                  'test acc: {:.6f}, real acc: {:.6f}'.format(
                    i, loss_val, accuracy_val, dev_accuracy, test_accuracy, real_accuracy))
            loss_list.append(loss_val)
            train_acc_list.append(accuracy_val)
            dev_acc_list.append(dev_accuracy)
            test_acc_list.append(test_accuracy)
            real_acc_list.append(real_accuracy)

    plt.subplot(121)
    plt.plot(loss_list, label=r'$Loss$')
    plt.legend(loc='best')
    plt.xlabel(r'$Iteration$')
    plt.ylabel(r'$Loss$')
    plt.grid()

    plt.subplot(122)
    plt.plot(dev_acc_list, label='验证集')
    plt.plot(test_acc_list, label='测试集')
    if not config.merge_real_into_test:
        plt.plot(move_avg(real_acc_list, config.real_move_avg), label='网络来源测试集')
    plt.legend(loc='best')
    plt.grid(which='both')
    plt.ylim(ymin=0.4, ymax=1.0)
    plt.xlabel(r'$Iteration$')
    plt.ylabel(r'$Accuracy$')

    plt.suptitle('RNN')

    plt.gcf().set_size_inches(12.80, 4.80)
    plt.savefig('PRLabCurves/result_rnn.png')
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    main()
    main_rnn()
    pass
