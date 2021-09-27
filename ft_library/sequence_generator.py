import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

class SequenceGenerator(QtWidgets.QDialog):
    update_sequence = QtCore.pyqtSignal(object, object, object)

    def __init__(self, experiment, settings, parent = None):
        QtWidgets.QDialog.__init__(self, parent)

        self.name = "Sequence Generator"
        self.settings = settings
        self.directory = self.settings['directory']
        self.header_sequence = experiment.header_sequence
        self.body_sequence = experiment.body_sequence
        self.footer_sequence = experiment.footer_sequence

        # configure UI
        import qt_designer.sequence_generator_ui as sequence_generator_ui

        self.ui = sequence_generator_ui.Ui_Sequence_Generator()
        self.ui.setupUi(self)

        # set title
        title = "Sequence Generator"
        self.setWindowTitle(title)

        # connect UI signals
        self.ui.pushButtonOpen.clicked.connect(self.open_sequence)
        self.ui.pushButtonSave.clicked.connect(self.save_sequence)
        self.ui.pushButtonClear.clicked.connect(self.clear_sequence)
        self.ui.pushButtonAdd.clicked.connect(self.add_frames)
        self.ui.radioButtonHeader.clicked.connect(self.edit_header)
        self.ui.radioButtonBody.clicked.connect(self.edit_body)
        self.ui.radioButtonFooter.clicked.connect(self.edit_footer)

        self.edit_header()

    def open_sequence(self):
        with open(self.directory + "header.txt", "r") as file:
            header_sequence = file.read()
            self.header_sequence = np.zeros((3, int(len(header_sequence) / 4)), dtype=np.int8)
            for n in np.arange(0, self.header_sequence.shape[1]):
                self.header_sequence[0, n] = header_sequence[4*n]
                self.header_sequence[1, n] = header_sequence[4*n+1]
                self.header_sequence[2, n] = header_sequence[4*n+2]
        with open(self.directory + "body.txt", "r") as file:
            body_sequence = file.read()
            self.body_sequence = np.zeros((3, int(len(body_sequence) / 4)), dtype=np.int8)
            for n in np.arange(0, self.body_sequence.shape[1]):
                self.body_sequence[0, n] = body_sequence[4 * n]
                self.body_sequence[1, n] = body_sequence[4 * n + 1]
                self.body_sequence[2, n] = body_sequence[4 * n + 2]
        with open(self.directory + "footer.txt", "r") as file:
            footer_sequence = file.read()
            self.footer_sequence = np.zeros((3, int(len(footer_sequence) / 4)), dtype=np.int8)
            for n in np.arange(0, self.footer_sequence.shape[1]):
                self.footer_sequence[0, n] = footer_sequence[4 * n]
                self.footer_sequence[1, n] = footer_sequence[4 * n + 1]
                self.footer_sequence[2, n] = footer_sequence[4 * n + 2]

        if self.button == 0:
            self.edit_header()
        elif self.button == 1:
            self.edit_body()
        elif self.button == 2:
            self.edit_footer()

        self.update_sequence.emit(self.header_sequence, self.body_sequence, self.footer_sequence)

    def save_sequence(self):
        with open(self.directory + "header.txt", "w") as file:
            for n in np.arange(0, np.shape(self.header_sequence)[1]):
                seq = self.header_sequence[:, n]
                file.write(f"{seq[0]}{seq[1]}{seq[2]}\n")
        with open(self.directory + "body.txt", "w") as file:
            for n in np.arange(0, np.shape(self.body_sequence)[1]):
                seq = self.body_sequence[:, n]
                file.write(f"{seq[0]}{seq[1]}{seq[2]}\n")
        with open(self.directory + "footer.txt", "w") as file:
            for n in np.arange(0, np.shape(self.footer_sequence)[1]):
                seq = self.footer_sequence[:, n]
                file.write(f"{seq[0]}{seq[1]}{seq[2]}\n")

    def clear_sequence(self):
        if self.button == 0:
            self.header_sequence = np.zeros((3, 0), dtype=np.int8)
        elif self.button == 1:
            self.body_sequence = np.zeros((3, 0), dtype=np.int8)
        elif self.button == 2:
            self.footer_sequence = np.zeros((3, 0), dtype=np.int8)

        self.ui.comboBoxSeq.clear()

        self.update_sequence.emit(self.header_sequence, self.body_sequence, self.footer_sequence)

    def add_frames(self):
        seq = np.array([1 if self.ui.checkBoxC1.isChecked() else 0, \
                        1 if self.ui.checkBoxC2.isChecked() else 0, \
                        1 if self.ui.checkBoxC3.isChecked() else 0])

        if np.sum(seq) != 0:
            if self.button == 0:
                for _ in np.arange(0, self.ui.spinBoxFrame.value()):
                    self.header_sequence = np.concatenate((self.header_sequence, np.resize(seq, (3, 1))), axis=1)
                    self.ui.comboBoxSeq.addItem \
                        (f"{self.ui.comboBoxSeq.count()}: \t {seq[0]}{seq[1]}{seq[2]}")
            elif self.button == 1:
                for _ in np.arange(0, self.ui.spinBoxFrame.value()):
                    self.body_sequence = np.concatenate((self.body_sequence, np.resize(seq, (3, 1))), axis=1)
                    self.ui.comboBoxSeq.addItem \
                        (f"{self.ui.comboBoxSeq.count()}: \t {seq[0]}{seq[1]}{seq[2]}")
            elif self.button == 2:
                for _ in np.arange(0, self.ui.spinBoxFrame.value()):
                    self.footer_sequence = np.concatenate((self.footer_sequence, np.resize(seq, (3, 1))), axis=1)
                    self.ui.comboBoxSeq.addItem\
                        (f"{self.ui.comboBoxSeq.count()}: \t {seq[0]}{seq[1]}{seq[2]}")
        else:
            print("select at least one channel")

        self.update_sequence.emit(self.header_sequence, self.body_sequence, self.footer_sequence)

    def edit_header(self):
        self.button = 0
        self.ui.radioButtonBody.setChecked(False)
        self.ui.radioButtonFooter.setChecked(False)

        self.ui.comboBoxSeq.clear()
        for n in np.arange(0, np.shape(self.header_sequence)[1]):
            seq = self.header_sequence[:, n]
            self.ui.comboBoxSeq.addItem \
                (f"{self.ui.comboBoxSeq.count()}: \t {seq[0]}{seq[1]}{seq[2]}")

    def edit_body(self):
        self.button = 1
        self.ui.radioButtonHeader.setChecked(False)
        self.ui.radioButtonFooter.setChecked(False)

        self.ui.comboBoxSeq.clear()
        for n in np.arange(0, np.shape(self.body_sequence)[1]):
            seq = self.body_sequence[:, n]
            self.ui.comboBoxSeq.addItem \
                (f"{self.ui.comboBoxSeq.count()}: \t {seq[0]}{seq[1]}{seq[2]}")

    def edit_footer(self):
        self.button = 2
        self.ui.radioButtonHeader.setChecked(False)
        self.ui.radioButtonBody.setChecked(False)

        self.ui.comboBoxSeq.clear()
        for n in np.arange(0, np.shape(self.footer_sequence)[1]):
            seq = self.footer_sequence[:, n]
            self.ui.comboBoxSeq.addItem \
                (f"{self.ui.comboBoxSeq.count()}: \t {seq[0]}{seq[1]}{seq[2]}")


