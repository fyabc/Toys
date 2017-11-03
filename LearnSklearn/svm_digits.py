#! /usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import datasets, svm, metrics

__author__ = 'fyabc'


def main():
    digits = datasets.load_digits()

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    classifier = svm.SVC(gamma=0.001)

    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])

    # print(expected[:10], predicted[:10])

    print('Confusion matrix:\n{}'.format(metrics.confusion_matrix(expected, predicted)))
    print('Classification report for classifier {}:\n{}'.format(
        classifier, metrics.classification_report(expected, predicted)))


if __name__ == '__main__':
    main()
