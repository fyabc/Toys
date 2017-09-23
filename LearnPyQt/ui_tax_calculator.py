# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tax_calculator.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TaxCalculator(object):
    def setupUi(self, TaxCalculator):
        TaxCalculator.setObjectName("TaxCalculator")
        TaxCalculator.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(TaxCalculator)
        self.centralwidget.setObjectName("centralwidget")
        self.price_box = QtWidgets.QTextEdit(self.centralwidget)
        self.price_box.setGeometry(QtCore.QRect(200, 110, 211, 41))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.price_box.setFont(font)
        self.price_box.setObjectName("price_box")
        self.label_price = QtWidgets.QLabel(self.centralwidget)
        self.label_price.setGeometry(QtCore.QRect(90, 120, 71, 21))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_price.setFont(font)
        self.label_price.setAlignment(QtCore.Qt.AlignCenter)
        self.label_price.setObjectName("label_price")
        self.tax_rate_box = QtWidgets.QSpinBox(self.centralwidget)
        self.tax_rate_box.setGeometry(QtCore.QRect(200, 193, 61, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.tax_rate_box.setFont(font)
        self.tax_rate_box.setProperty("value", 20)
        self.tax_rate_box.setObjectName("tax_rate_box")
        self.label_tax_rate = QtWidgets.QLabel(self.centralwidget)
        self.label_tax_rate.setGeometry(QtCore.QRect(90, 200, 91, 21))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_tax_rate.setFont(font)
        self.label_tax_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tax_rate.setObjectName("label_tax_rate")
        self.calc_tax_button = QtWidgets.QPushButton(self.centralwidget)
        self.calc_tax_button.setGeometry(QtCore.QRect(200, 270, 141, 31))
        self.calc_tax_button.setObjectName("calc_tax_button")
        self.result_box = QtWidgets.QTextEdit(self.centralwidget)
        self.result_box.setGeometry(QtCore.QRect(200, 370, 211, 41))
        self.result_box.setReadOnly(True)
        self.result_box.setObjectName("result_box")
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        self.label_title.setGeometry(QtCore.QRect(170, 20, 431, 71))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setObjectName("label_title")
        TaxCalculator.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TaxCalculator)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        TaxCalculator.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(TaxCalculator)
        self.statusbar.setObjectName("statusbar")
        TaxCalculator.setStatusBar(self.statusbar)

        self.retranslateUi(TaxCalculator)
        QtCore.QMetaObject.connectSlotsByName(TaxCalculator)

    def retranslateUi(self, TaxCalculator):
        _translate = QtCore.QCoreApplication.translate
        TaxCalculator.setWindowTitle(_translate("TaxCalculator", "MainWindow"))
        self.label_price.setText(_translate("TaxCalculator", "Price"))
        self.label_tax_rate.setText(_translate("TaxCalculator", "Tax Rate"))
        self.calc_tax_button.setText(_translate("TaxCalculator", "Calculate Tax"))
        self.label_title.setText(_translate("TaxCalculator", "Sales Tax Calculator"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TaxCalculator = QtWidgets.QMainWindow()
    ui = Ui_TaxCalculator()
    ui.setupUi(TaxCalculator)
    TaxCalculator.show()
    sys.exit(app.exec_())

