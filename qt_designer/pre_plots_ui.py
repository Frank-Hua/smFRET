# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\frank\Documents\fret_tools3\qt_designer\pre_plots_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(430, 70)
        Form.setMinimumSize(QtCore.QSize(430, 70))
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.doubleSpinBoxIntensityCutoff = QtWidgets.QDoubleSpinBox(Form)
        self.doubleSpinBoxIntensityCutoff.setWrapping(False)
        self.doubleSpinBoxIntensityCutoff.setDecimals(1)
        self.doubleSpinBoxIntensityCutoff.setMinimum(-100000.0)
        self.doubleSpinBoxIntensityCutoff.setMaximum(100000.0)
        self.doubleSpinBoxIntensityCutoff.setObjectName("doubleSpinBoxIntensityCutoff")
        self.horizontalLayout.addWidget(self.doubleSpinBoxIntensityCutoff)
        self.pushButtonApply = QtWidgets.QPushButton(Form)
        self.pushButtonApply.setObjectName("pushButtonApply")
        self.horizontalLayout.addWidget(self.pushButtonApply)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Currunt Intensity Cutoff:"))
        self.pushButtonApply.setText(_translate("Form", "Apply"))
