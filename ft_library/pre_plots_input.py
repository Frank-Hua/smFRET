from PyQt5 import QtCore, QtGui, QtWidgets

class PrePlotsInput(QtWidgets.QDialog):
    update_intensity_cutoff = QtCore.pyqtSignal(float)

    def __init__(self, parent = None):
        QtWidgets.QDialog.__init__(self, parent)

        self.name = "Pre-Plots Input"
        self.intensity_cutoff = 0

        # configure UI
        import qt_designer.pre_plots_ui as pre_plots_ui

        self.ui = pre_plots_ui.Ui_Form()
        self.ui.setupUi(self)

        # set title
        title = "Pre-Plots Input"
        self.setWindowTitle(title)

        self.ui.doubleSpinBoxIntensityCutoff.setValue(self.intensity_cutoff)

        # connect UI signals
        self.ui.pushButtonApply.clicked.connect(self.set_intensity_cutoff)

    def set_intensity_cutoff(self):
        self.intensity_cutoff = self.ui.doubleSpinBoxIntensityCutoff.value()
        self.update_intensity_cutoff.emit(self.intensity_cutoff)


