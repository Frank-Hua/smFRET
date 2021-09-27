#!/usr/bin/python

# load the appropriate application icon

from PyQt5 import QtCore, QtGui, QtWidgets

class QAppIcon(QtGui.QIcon):
    def __init__(self):
        QtGui.QIcon.__init__(self, "./icons/fret.ico")


