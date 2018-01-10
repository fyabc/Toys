#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtWidgets, uic

TYPED = False

# Two ways to import UI file:
if TYPED:
    # 1. This need to use pyuic5 to compile the UI file, but has PyCharm type support.
    # $ pyuic5 tax_calculator.ui -o ui_tax_calculator.py
    try:
        from .ui_tax_calculator import Ui_TaxCalculator
    except SystemError:
        from ui_tax_calculator import Ui_TaxCalculator
else:
    # 2. This does not need to compile, but does not have PyCharm type support.
    Ui_TaxCalculator, _ = uic.loadUiType('./tax_calculator.ui')

__author__ = 'fyabc'


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_TaxCalculator()
        self.ui.setupUi(self)

        self.ui.calc_tax_button.clicked.connect(self.calculate_tax)

    def calculate_tax(self):
        try:
            price = int(self.ui.price_box.toPlainText())
        except ValueError:
            return

        tax_rate = self.ui.tax_rate_box.value()
        total_price = price * (1 + tax_rate / 100.0)

        self.ui.result_box.setText('Total price with tax is: {:.2f}'.format(total_price))


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
