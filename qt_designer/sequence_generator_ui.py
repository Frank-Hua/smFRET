# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\frank\Documents\fret_tools3\qt_designer\sequence_generator_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Sequence_Generator(object):
    def setupUi(self, Sequence_Generator):
        Sequence_Generator.setObjectName("Sequence_Generator")
        Sequence_Generator.resize(850, 170)
        Sequence_Generator.setMinimumSize(QtCore.QSize(850, 170))
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(Sequence_Generator)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame = QtWidgets.QFrame(Sequence_Generator)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.radioButtonHeader = QtWidgets.QRadioButton(self.frame)
        self.radioButtonHeader.setMinimumSize(QtCore.QSize(0, 34))
        self.radioButtonHeader.setChecked(True)
        self.radioButtonHeader.setObjectName("radioButtonHeader")
        self.verticalLayout_2.addWidget(self.radioButtonHeader)
        self.radioButtonBody = QtWidgets.QRadioButton(self.frame)
        self.radioButtonBody.setMinimumSize(QtCore.QSize(0, 34))
        self.radioButtonBody.setObjectName("radioButtonBody")
        self.verticalLayout_2.addWidget(self.radioButtonBody)
        self.radioButtonFooter = QtWidgets.QRadioButton(self.frame)
        self.radioButtonFooter.setMinimumSize(QtCore.QSize(0, 34))
        self.radioButtonFooter.setObjectName("radioButtonFooter")
        self.verticalLayout_2.addWidget(self.radioButtonFooter)
        self.horizontalLayout_4.addWidget(self.frame)
        self.frame_4 = QtWidgets.QFrame(Sequence_Generator)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.comboBoxSeq = QtWidgets.QComboBox(self.frame_4)
        self.comboBoxSeq.setObjectName("comboBoxSeq")
        self.verticalLayout_6.addWidget(self.comboBoxSeq)
        self.frame_7 = QtWidgets.QFrame(self.frame_4)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButtonOpen = QtWidgets.QPushButton(self.frame_7)
        self.pushButtonOpen.setObjectName("pushButtonOpen")
        self.horizontalLayout_2.addWidget(self.pushButtonOpen)
        self.pushButtonSave = QtWidgets.QPushButton(self.frame_7)
        self.pushButtonSave.setObjectName("pushButtonSave")
        self.horizontalLayout_2.addWidget(self.pushButtonSave)
        self.pushButtonClear = QtWidgets.QPushButton(self.frame_7)
        self.pushButtonClear.setObjectName("pushButtonClear")
        self.horizontalLayout_2.addWidget(self.pushButtonClear)
        self.pushButtonAdd = QtWidgets.QPushButton(self.frame_7)
        self.pushButtonAdd.setObjectName("pushButtonAdd")
        self.horizontalLayout_2.addWidget(self.pushButtonAdd)
        self.spinBoxFrame = QtWidgets.QSpinBox(self.frame_7)
        self.spinBoxFrame.setMinimum(1)
        self.spinBoxFrame.setMaximum(1000)
        self.spinBoxFrame.setObjectName("spinBoxFrame")
        self.horizontalLayout_2.addWidget(self.spinBoxFrame)
        self.label = QtWidgets.QLabel(self.frame_7)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.verticalLayout_6.addWidget(self.frame_7)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem)
        self.horizontalLayout_4.addWidget(self.frame_4)
        self.frame_2 = QtWidgets.QFrame(Sequence_Generator)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.checkBoxC1 = QtWidgets.QCheckBox(self.frame_2)
        self.checkBoxC1.setObjectName("checkBoxC1")
        self.verticalLayout.addWidget(self.checkBoxC1)
        self.checkBoxC2 = QtWidgets.QCheckBox(self.frame_2)
        self.checkBoxC2.setObjectName("checkBoxC2")
        self.verticalLayout.addWidget(self.checkBoxC2)
        self.checkBoxC3 = QtWidgets.QCheckBox(self.frame_2)
        self.checkBoxC3.setObjectName("checkBoxC3")
        self.verticalLayout.addWidget(self.checkBoxC3)
        self.horizontalLayout_4.addWidget(self.frame_2)

        self.retranslateUi(Sequence_Generator)
        self.comboBoxSeq.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(Sequence_Generator)

    def retranslateUi(self, Sequence_Generator):
        _translate = QtCore.QCoreApplication.translate
        Sequence_Generator.setWindowTitle(_translate("Sequence_Generator", "Form"))
        self.radioButtonHeader.setText(_translate("Sequence_Generator", "header"))
        self.radioButtonBody.setText(_translate("Sequence_Generator", "body"))
        self.radioButtonFooter.setText(_translate("Sequence_Generator", "footer"))
        self.pushButtonOpen.setText(_translate("Sequence_Generator", "Open"))
        self.pushButtonSave.setText(_translate("Sequence_Generator", "Save"))
        self.pushButtonClear.setText(_translate("Sequence_Generator", "Clear"))
        self.pushButtonAdd.setText(_translate("Sequence_Generator", "Add"))
        self.label.setText(_translate("Sequence_Generator", "Frame(s)"))
        self.checkBoxC1.setText(_translate("Sequence_Generator", "Channel 1"))
        self.checkBoxC2.setText(_translate("Sequence_Generator", "Channel 2"))
        self.checkBoxC3.setText(_translate("Sequence_Generator", "Channel 3"))