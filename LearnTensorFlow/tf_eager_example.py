#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Example of TensorFlow eager execution."""

import tensorflow as tf

tf.enable_eager_execution()
tfe = tf.contrib.eager


class MyDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel', [input_shape[-1], self.units])
        self.bias = self.add_variable('bias', [self.units])

        super().build(input_shape)

    def call(self, x):
        return tf.add(tf.matmul(x, self.kernel), self.bias)


class SingleTest(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(1.)
        self.b = tf.Variable(0.)

    def call(self, x):
        return self.w * x + self.b


def make_dataset_1(size=2000):
    # Target: y = 3x + 2
    x = tf.random_normal([size])
    noise = tf.random_normal([size], stddev=0.1)
    y = x * 3 + 2 + noise
    dataset = tf.data.Dataset.from_tensor_slices({
        'x': x,
        'y': y,
    })
    dataset = dataset.batch(20)
    return dataset


def loss_1(model, x, y):
    error = model(x) - y
    return tf.reduce_mean(tf.square(error))


def grad_1(model: tf.keras.Model, x, y):
    with tf.GradientTape() as tape:
        loss = loss_1(model, x, y)
    return tape.gradient(loss, model.variables)


def main():
    dataset = make_dataset_1()
    model = SingleTest()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    for _ in range(3):
        for i, data in enumerate(dataset):
            grads = grad_1(model, data['x'], data['y'])
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

            if i % 10 == 0:
                print('Step {} loss: {:.3f}'.format(i, loss_1(model, data['x'], data['y'])))
                print(model.w.numpy(), model.b.numpy())


if __name__ == '__main__':
    main()
