#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Try TensorFlow `Experiment` and use it with flexible input."""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.contrib.training import HParams
from tensorflow.contrib.learn import learn_runner, RunConfig, Experiment
from tensorflow.contrib import slim, layers
from tensorflow.contrib.data import Dataset

__author__ = 'fyabc'

# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)

# Set default flags for the output directories
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    flag_name='model_dir', default_value='./output/MNIST_experiment',
    docstring='Output directory for model and training stats.')
flags.DEFINE_string(
    flag_name='data_dir', default_value='./MNIST_data',
    docstring='Directory to download the data to.')
flags.DEFINE_integer(
    flag_name='batch_size', default_value=128,
    docstring='Batch size. [128]')
flags.DEFINE_integer(
    flag_name='min_eval_frequency', default_value=500,
    docstring='Minimum evaluate frequency. [500]')
flags.DEFINE_boolean(
    flag_name='which_input', default_value=True,
    docstring='Which input to use, False = dataset, True = manual input. [True]'
)
flags.DEFINE_boolean(
    flag_name='hook2', default_value=False,
    docstring='Use `FeedInputHook2`. [False]'
)


def _collection_name(is_training):
    _s = 'TrainingData' if is_training else 'TestData'
    return '{}/images'.format(_s), '{}/labels'.format(_s)


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialize data iterator after Session is created."""

    def __init__(self):
        super().__init__()
        self.iterator_initializer_func = None

        self.num_batches = 0

    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created."""
        self.iterator_initializer_func(session)

    def before_run(self, run_context: tf.train.SessionRunContext):
        self.num_batches += 1


class FeedInputHook(tf.train.SessionRunHook):
    """Read data manually, then feed it into placeholders."""
    def __init__(self, images, labels, batch_size, is_training):
        super(FeedInputHook, self).__init__()
        self.images = images
        self.labels = labels
        self.data_size = len(self.labels)
        self.batch_size = batch_size
        self.is_training = is_training
        self.n_batches = 0

        self.next_sample_ph = None
        self.next_labels_ph = None

    def make_placeholder(self):
        with tf.name_scope('TrainingData' if self.is_training else 'TestData'):
            next_sample = tf.placeholder(dtype=self.images.dtype, shape=[None, 28, 28, 1], name='next_sample')
            next_labels = tf.placeholder(dtype=self.labels.dtype, shape=[None], name='next_label')

            self.next_sample_ph = next_sample
            self.next_labels_ph = next_labels

        return next_sample, next_labels

    def begin(self):
        # Reset the counter in testing mode.
        # [NOTE] Shall I also reset it in training mode?
        if not self.is_training:
            self.n_batches = 0

    def before_run(self, run_context: tf.train.SessionRunContext):
        if self.is_training:
            batch_start = (self.n_batches * self.batch_size) % self.data_size
            batch_end = batch_start + self.batch_size

            if batch_end <= self.data_size:
                batch_images, batch_labels = self.images[batch_start:batch_end], self.labels[batch_start:batch_end]
            else:
                batch_end %= self.data_size
                batch_images = np.concatenate([self.images[batch_start:], self.images[:batch_end]], axis=0)
                batch_labels = np.concatenate([self.labels[batch_start:], self.labels[:batch_end]], axis=0)
        else:
            batch_start = self.n_batches * self.batch_size
            batch_end = batch_start + self.batch_size
            batch_images, batch_labels = self.images[batch_start:batch_end], self.labels[batch_start:batch_end]

        next_sample = self.next_sample_ph
        next_labels = self.next_labels_ph

        self.n_batches += 1

        # Add next batch data into feed dict.
        orig_feed_dict = run_context.original_args.feed_dict
        if orig_feed_dict is None:
            orig_feed_dict = {}
        else:
            orig_feed_dict = orig_feed_dict.copy()
        orig_feed_dict.update({
            next_sample: batch_images,
            next_labels: batch_labels,
        })

        return tf.train.SessionRunArgs(
            fetches=run_context.original_args.fetches,
            feed_dict=orig_feed_dict,
            options=run_context.original_args.options,
        )

    def after_run(self, run_context, run_values):
        # In test mode, stop after one epoch.
        if not self.is_training and self.batch_size * self.n_batches >= self.data_size:
            run_context.request_stop()

        # [NOTE]: loss and other variables are stored in `run_values`.
        # The hook can get them to implement some algorithms, such as self-paced learning.


class FeedInputHook2(tf.train.SessionRunHook):
    """Read data from `Dataset`, then feed it into placeholders."""
    def __init__(self, images, labels, batch_size, is_training):
        super(FeedInputHook2, self).__init__()
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.is_training = is_training
        self.n_batches = 0

        self.next_sample = None
        self.next_labels = None
        self.next_sample_ph = None
        self.next_labels_ph = None
        self.init_dataset_fn = None

    def after_create_session(self, session, coord):
        self.init_dataset_fn(session)

    def make_placeholder(self):
        with tf.name_scope('TrainingData' if self.is_training else 'TestData'):
            images_placeholder = tf.placeholder(self.images.dtype, self.images.shape, name='images')
            labels_placeholder = tf.placeholder(self.labels.dtype, self.labels.shape, name='labels')

            # [NOTE]: Must use tuple here
            dataset = Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
            if self.is_training:
                dataset = dataset.repeat(None)  # Infinite iterations
                dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()

            # TrainingData/IteratorGetNext:0, TrainingData/IteratorGetNext:1
            self.next_sample, self.next_labels = iterator.get_next()

            # Make placeholders
            self.next_sample_ph = tf.placeholder(
                dtype=self.next_sample.dtype, shape=self.next_sample.shape, name='next_sample')
            self.next_labels_ph = tf.placeholder(
                dtype=self.next_labels.dtype, shape=self.next_labels.shape, name='next_label')

            self.init_dataset_fn = lambda session: session.run(
                iterator.initializer,
                feed_dict={
                    images_placeholder: self.images,
                    labels_placeholder: self.labels,
                }
            )

        return self.next_sample_ph, self.next_labels_ph

    def begin(self):
        # Reset the counter in testing mode.
        # [NOTE] Shall I also reset it in training mode?
        if not self.is_training:
            self.n_batches = 0

    def before_run(self, run_context: tf.train.SessionRunContext):
        batch_images, batch_labels = run_context.session.run([self.next_sample, self.next_labels])

        self.n_batches += 1

        # Add next batch data into feed dict.
        orig_feed_dict = run_context.original_args.feed_dict
        if orig_feed_dict is None:
            orig_feed_dict = {}
        else:
            orig_feed_dict = orig_feed_dict.copy()
        orig_feed_dict.update({
            self.next_sample_ph: batch_images,
            self.next_labels_ph: batch_labels,
        })

        return tf.train.SessionRunArgs(
            fetches=run_context.original_args.fetches,
            feed_dict=orig_feed_dict,
            options=run_context.original_args.options,
        )

    def after_run(self, run_context, run_values):
        pass


def get_input2(batch_size: int, mnist_data, mode, **kwargs):
    is_training = mode == ModeKeys.TRAIN

    if is_training:
        images = mnist_data.train.images.reshape([-1, 28, 28, 1])
        labels = mnist_data.train.labels.astype('int32')
    else:
        images = mnist_data.test.images.reshape([-1, 28, 28, 1])
        labels = mnist_data.test.labels.astype('int32')

    if kwargs.pop('hook2', False) is True:
        hook_class = FeedInputHook2
    else:
        hook_class = FeedInputHook
    feed_input_hook = hook_class(images, labels, batch_size, is_training)

    def input_fn():
        return feed_input_hook.make_placeholder()

    return input_fn, feed_input_hook


def get_input(batch_size: int, mnist_data, mode, **kwargs):
    is_training = mode == ModeKeys.TRAIN

    iterator_initializer_hook = IteratorInitializerHook()

    if is_training:
        images = mnist_data.train.images.reshape([-1, 28, 28, 1])
        labels = mnist_data.train.labels.astype('int32')
    else:
        images = mnist_data.test.images.reshape([-1, 28, 28, 1])
        labels = mnist_data.test.labels.astype('int32')

    def input_fn():
        with tf.name_scope('TrainingData' if is_training else 'TestData'):
            images_placeholder = tf.placeholder(images.dtype, images.shape, name='images')
            labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name='labels')

            # [NOTE]: Must use tuple here
            dataset = Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
            if is_training:
                dataset = dataset.repeat(None)  # Infinite iterations
                dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()

            # TrainingData/IteratorGetNext:0, TrainingData/IteratorGetNext:1
            next_sample, next_label = iterator.get_next()

            # Session run hook to initialize iterator with data
            iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(
                iterator.initializer,
                feed_dict={
                    images_placeholder: images,
                    labels_placeholder: labels,
                }
            )

            return next_sample, next_label
    return input_fn, iterator_initializer_hook


def model_main(inputs, is_training=True, scope='MnistConvNet'):
    """The main architecture of the model.

    :param inputs:
    :param is_training:
    :param scope: Variable scope name.
    :return: logits
    """

    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer()):
            net = slim.conv2d(inputs, 20, [5, 5], padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')

            net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='conv3')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool4')

            net = tf.reshape(net, [-1, 4 * 4 * 40])
            net = slim.fully_connected(net, 256, scope='fn5')
            net = slim.dropout(net, is_training=is_training, scope='dropout5')
            net = slim.fully_connected(net, 256, scope='fn6')
            net = slim.dropout(net, is_training=is_training, scope='dropout6')

            net = slim.fully_connected(net, 10, scope='output', activation_fn=None)

        return net


def model_fn(features, labels, mode, params):
    is_training = mode == ModeKeys.TRAIN

    logits = model_main(features, is_training=is_training)
    predictions = tf.argmax(logits, axis=-1)

    loss = None
    train_op = None
    eval_metric_ops = {}

    if mode != ModeKeys.PREDICT:
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits,
        )
        train_op = layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=params.learning_rate,
            name='train_op',
        )
        eval_metric_ops = {
            'Accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                name='accuracy',
            )
        }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
    )


def experiment_fn(run_config: RunConfig, params: HParams):
    """The experiment function, provided to learn_runner.

    :param run_config:
    :param params:
    :return:
    """

    run_config = run_config.replace(save_checkpoints_steps=params.min_eval_frequency)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config,
    )

    mnist_data = input_data.read_data_sets(train_dir='MNIST_data/', one_hot=False)
    if FLAGS.which_input:
        get_input_fn = get_input2
    else:
        get_input_fn = get_input
    train_input_fn, train_input_hook = get_input_fn(
        FLAGS.batch_size, mnist_data, mode=ModeKeys.TRAIN, hook2=FLAGS.hook2)
    eval_input_fn, eval_input_hook = get_input_fn(
        FLAGS.batch_size, mnist_data, mode=ModeKeys.EVAL, hook2=FLAGS.hook2)

    experiment = Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=params.train_steps,
        min_eval_frequency=params.min_eval_frequency,
        train_monitors=[train_input_hook],
        eval_hooks=[eval_input_hook],
        eval_steps=None,
    )

    return experiment


def main(_):
    """Run the experiment."""

    # Define model hyper-parameters
    params = HParams(
        learning_rate=0.002,
        n_classes=10,
        train_steps=5000,
        min_eval_frequency=FLAGS.min_eval_frequency,
    )

    run_config = RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule='train_and_evaluate',
        hparams=params,
    )


if __name__ == '__main__':
    tf.app.run()
