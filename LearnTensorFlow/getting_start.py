#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

__author__ = 'fyabc'


def simple_linear_model():
    se = tf.Session()

    W = tf.Variable([.3], tf.float32, name='W')
    b = tf.Variable([-.3], tf.float32, name='b')
    x = tf.placeholder(tf.float32, name='x')
    linear_model = W * x + b

    init = tf.global_variables_initializer()

    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    # Variables must be initialized.
    se.run(init)

    y = tf.placeholder(tf.float32)

    feed_dict = {x: x_train, y: y_train}

    print(se.run(linear_model, feed_dict))

    square_deltas = (linear_model - y) ** 2
    loss = tf.reduce_sum(square_deltas, name='loss')

    print('Init:', se.run(loss, feed_dict))

    fix_W = tf.assign(W, [-1.])
    fix_b = tf.assign(b, [1.])
    se.run([fix_W, fix_b])
    print('After assign:', se.run(loss, feed_dict))

    # Usage of tf.train.
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss, name='train')

    se.run(init)    # Reset to incorrect values

    iteration = 1000
    for _ in range(iteration):
        se.run(train, feed_dict)

    curr_W, curr_b, curr_loss = se.run([W, b, loss], feed_dict)

    print('After train {} iterations: W = {}, b = {}, loss = {}'.format(iteration, curr_W, curr_b, curr_loss))


def learn_basic():
    # Declare list of features. We only have one real-valued feature. There are many
    # other types of columns that are more complicated and useful.
    features = [tf.contrib.layers.real_valued_column('x', dimension=1)]

    # An estimator is the front end to invoke training (fitting) and evaluation
    # (inference). There are many predefined types like linear regression,
    # logistic regression, linear classification, logistic classification, and
    # many neural network classifiers and regressors. The following code
    # provides an estimator that does linear regression.
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

    # TensorFlow provides many helper methods to read and set up data sets.
    # Here we use `numpy_input_fn`. We have to tell the function how many batches
    # of data (num_epochs) we want and how big each batch should be.
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({'x': x}, y, batch_size=4, num_epochs=1000)

    # We can invoke 1000 training steps by invoking the `fit` method and passing the
    # training data set.
    estimator.fit(input_fn=input_fn, steps=1000)

    # Here we evaluate how well our model did. In a real example, we would want
    # to use a separate validation and testing data set to avoid overfitting.
    print(estimator.evaluate(input_fn=input_fn))


def learn_custom():
    # Declare list of features, we only have one real-valued feature
    def model(features, labels, mode):
        # Build a linear model and predict values
        W = tf.get_variable('W', [1], dtype=tf.float64)
        b = tf.get_variable('b', [1], dtype=tf.float64)
        y = W * features['x'] + b
        # Loss sub-graph
        loss = tf.reduce_sum(tf.square(y - labels))
        # Training sub-graph
        global_step = tf.train.get_global_step()
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = tf.group(optimizer.minimize(loss),
                         tf.assign_add(global_step, 1))
        # ModelFnOps connects sub-graphs we built to the
        # appropriate functionality.
        return tf.contrib.learn.ModelFnOps(
            mode=mode, predictions=y,
            loss=loss,
            train_op=train)

    estimator = tf.contrib.learn.Estimator(model_fn=model)
    # define our data set
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

    # train
    estimator.fit(input_fn=input_fn, steps=1000)
    # evaluate our model
    print(estimator.evaluate(input_fn=input_fn, steps=10))


def main():
    # simple_linear_model()
    # learn_basic()
    learn_custom()

    pass


if __name__ == '__main__':
    main()
