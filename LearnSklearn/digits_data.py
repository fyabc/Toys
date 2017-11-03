#! /usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import datasets

__author__ = 'fyabc'


digits = datasets.load_digits()


def show_contents():
    for k, v in digits.items():
        try:
            print(k, v.shape)
        except:
            print(k, v)


def show_images():
    images_labels = list(zip(digits.images, digits.target))
    for i, (x, y) in enumerate(images_labels[:4]):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.imshow(x, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training {}'.format(y))

    plt.show()


def main():
    show_contents()
    show_images()


if __name__ == '__main__':
    main()
