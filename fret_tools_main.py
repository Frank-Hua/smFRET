__author__ = 'frank hua'

import os
import sys
import argparse

import math
import numpy as np
from scipy import signal

import pickle

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5 import QtCore, QtGui, QtWidgets

from trajectory import Trajectory
from movie import Movie
from experiment import Experiment
import ft_library.fret_settings as fret_settings
import ft_library.fitting_tools as fitting_tools
import ft_library.pre_plots_input as pre_plots_input
import ft_library.sequence_generator as sequence_generator

def analyze_raw(settings):
    print("perform raw analysis")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file")
    parser.add_argument("--make-trajectories",
                        help="Makes trajectories",
                        action='store_true')
    parser.add_argument("--new-analysis",
                        help="New analysis",
                        action='store_true')
    '''
    parser.add_argument("--threads",
                        help="Max number of processes to use",
                        type = int, default = 8)
    '''
    args = parser.parse_args()

    if args.make_trajectories:
        args.new_analysis = True

    return args

class Window(QtWidgets.QMainWindow):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        args = parse_args()
        self.settings = fret_settings.fret_settings(args.settings_file)
        self.name = self.settings['experiment_name'] + '.pkl'
        self.directory = self.settings['directory']

        if args.make_trajectories:
            print('select spots and make trajectories')
            analyze_raw(self.settings)
            print('...')
            print('done selecting spots and making trajectories')

        if args.new_analysis:
            self.experiment = Experiment(self.settings)
        else:
            self.experiment = self.unpickle()
        print('experiment ready')

        # configure UI
        import qt_designer.fret_tools_ui as fret_tools_ui
        import qt_widgets.qt_app_icon as qt_app_icon

        self.ui = fret_tools_ui.Ui_MainWindow()
        self.ui.setupUi(self)

        # set icon
        self.setWindowIcon(qt_app_icon.QAppIcon())

        # set title
        title = self.experiment.settings["experiment_name"]
        self.setWindowTitle(title)

        self.current_movie_num = 0
        self.current_movie = self.experiment.movies[0]
        self.current_trajectory_num = 0
        self.current_trajectory = self.experiment.movies[0].trajectories[0]

        self.ui.spinBoxMovie.setMaximum(self.experiment.movie_num-1)
        self.apply_to_experiment = self.ui.checkBoxApplyToAllMovies.isChecked()
        self.skip_trajectory = self.ui.checkBoxSkipTrajectories.isChecked()
        self.mid = 0
        self.data_points = None
        self.modules = []

        # connect UI signals
        self.ui.action_Plot_Intensity_Histogram.triggered.connect(self.plot_intensity_histogram)
        self.ui.action_Make_E_S_Plot.triggered.connect(self.make_e_s_plot)
        self.ui.action_Plot_FRET_Histogram_Overall.triggered.connect(self.plot_fret_histogram_overall)
        self.ui.action_Plot_FRET_Histogram.triggered.connect(self.plot_fret_histogram)
        self.ui.action_Plot_Cross_Correlation.triggered.connect(self.plot_cross_correlation)
        self.ui.action_Close.triggered.connect(self.close_window)
        self.ui.action_Edit_Sequence.triggered.connect(self.edit_sequence)
        self.ui.spinBoxMovie.valueChanged.connect(self.goto_movie)
        self.ui.checkBoxMovie.stateChanged.connect(self.toggle_movie)
        self.ui.checkBoxApplyToAllMovies.stateChanged.connect(self.toggle_apply_to_experiment)
        self.ui.buttonBoxCorrections.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply_correction)
        self.ui.buttonBoxCorrections.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(self.reset_correction)
        self.ui.spinBoxTrajectory.valueChanged.connect(self.goto_trajectory)
        self.ui.checkBoxSkipTrajectories.stateChanged.connect(self.toggle_skip_trajectory)
        self.ui.forwardButton.clicked.connect(self.next_trajectory)
        self.ui.backButton.clicked.connect(self.previous_trajectory)
        self.ui.checkBoxTrajectory.stateChanged.connect(self.toggle_trajectory)
        self.ui.spinBoxCategory.valueChanged.connect(self.set_category)
        self.ui.toolButtonClickPoints.clicked.connect(self.click_data_points)
        self.ui.doubleSpinBoxTimeOffset.valueChanged.connect(self.update_time_offset)
        self.ui.pushButtonResetTrajectory.clicked.connect(self.reset_trajectory)
        self.ui.pushButtonResetMovie.clicked.connect(self.reset_movie)
        self.ui.saveTrajectoryButton.clicked.connect(self.save_trajectory)
        self.ui.pushButtonClearTrajectory.clicked.connect(self.clear_trajectory)
        self.ui.pushButtonClearMovie.clicked.connect(self.clear_movie)
        self.ui.pushButtonSavePoints.clicked.connect(self.save_data_points)
        self.ui.comboBoxPlotSide.currentIndexChanged.connect(self.plot_side)

        # initialize the main figure
        plt.ion()
        self.figure = plt.figure(dpi=200, facecolor='0.85')
        self.axes = self.figure.subplots(2, 1, sharex=True)
        plt.close(self.figure)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ui.verticalLayout_plot.addWidget(self.toolbar)
        self.ui.verticalLayout_plot.addWidget(self.canvas)

        # initialize the side figure
        self.figure_side = plt.figure(dpi=200, facecolor='0.85')
        self.axis_side = self.figure_side.subplots(1, 1)
        plt.close(self.figure_side)
        self.canvas_side = FigureCanvas(self.figure_side)
        self.ui.verticalLayout_plot_side.addWidget(self.canvas_side)

        self.update_ui()

    def unpickle(self):
        with open(self.directory + self.name, "rb") as file:
            one_experiment = pickle.load(file)
        return one_experiment

    def update_ui(self):
        self.plot_trajectory()
        self.plot_side()
        if self.current_movie.movie_included:
            self.ui.checkBoxMovie.setChecked(True)
        else:
            self.ui.checkBoxMovie.setChecked(False)
        self.ui.spinBoxTrajectory.setValue(self.current_trajectory_num)
        self.ui.spinBoxTrajectory.setMaximum(self.current_movie.trajectory_num - 1)

        correction = self.current_trajectory.correction
        self.ui.doubleSpinBoxDdBsl.setValue(correction[0])
        self.ui.doubleSpinBoxAdBsl.setValue(correction[1])
        self.ui.doubleSpinBoxAaBsl.setValue(correction[2])
        self.ui.doubleSpinBoxReflection.setValue(correction[3])
        self.ui.doubleSpinBoxDonorLeakage.setValue(correction[4])
        self.ui.doubleSpinBoxDirectEx.setValue(correction[5])
        self.ui.doubleSpinBoxDetection.setValue(correction[6])
        self.ui.doubleSpinBoxNormalizedEx.setValue(correction[7])
        self.ui.doubleSpinBoxTimeOffset.setValue(self.current_trajectory.time_offset)

        if self.current_trajectory.trajectory_included:
            self.ui.checkBoxTrajectory.setChecked(True)
            self.ui.spinBoxCategory.setValue(self.current_trajectory.category)
            self.ui.spinBoxCategory.setEnabled(True)
        else:
            self.ui.checkBoxTrajectory.setChecked(False)
            self.ui.spinBoxCategory.setValue(self.current_trajectory.category)
            self.ui.spinBoxCategory.setEnabled(False)

        self.ui.comboBoxSelectedPoints.clear()
        if self.current_trajectory.data_points is not None:
            for data_point in self.current_trajectory.data_points:
                self.ui.comboBoxSelectedPoints.addItem\
                    (f"x={data_point[0]:.2f} \t y={data_point[1]:.2f} \t {data_point[2]}")

    def plot_trajectory(self):
        # print("plot movie "+str(self.current_movie_num)+" trajectory "+str(self.current_trajectory_num))
        self.current_trajectory.plot_trajectory(axes=self.axes)

    def plot_side(self):
        if self.ui.comboBoxPlotSide.currentIndex() == 0:
            self.axis_side.clear()
            self.current_trajectory.plot_fret_histogram(axis=self.axis_side)
        elif self.ui.comboBoxPlotSide.currentIndex() == 1:
            self.axis_side.clear()
            self.current_trajectory.plot_x_correlation(axis=self.axis_side)
        elif self.ui.comboBoxPlotSide.currentIndex() == 2:
            self.axis_side.clear()
            print("show the raw image")

    def plot_intensity_histogram(self):
        # print("plot intensity histogram")
        self.experiment.make_intensity_histogram()

    def make_e_s_plot(self):
        # print("make E-S plot")
        if "Pre-Plots Input" not in [module.name for module in self.modules]:
            pre_plots_input_dialog = pre_plots_input.PrePlotsInput()
            pre_plots_input_dialog.update_intensity_cutoff.connect(self.experiment.make_pre_plots)
            self.modules.append(pre_plots_input_dialog)
            self.experiment.make_pre_plots(pre_plots_input_dialog.intensity_cutoff)
        else:
            pre_plots_input_dialog = self.modules[[module.name for module in self.modules].index("Pre-Plots Input")]
        pre_plots_input_dialog.show()
        self.experiment.make_e_s_2dhistogram()

    def plot_fret_histogram_overall(self):
        # print("plot FRET histogram")
        self.experiment.make_fret_histogram()

    def plot_fret_histogram(self):
        self.experiment.calculate_fret_histogram()

    def plot_cross_correlation(self):
        # ("plot cross-correlation")
        self.experiment.calculate_cross_correlation()

    def close_window(self):
        self.experiment.make_pickle()
        self.close()

    def edit_sequence(self):
        # print("edit sequence")
        if "Sequence Generator" not in [module.name for module in self.modules]:
            sequence_generator_dialog = sequence_generator.SequenceGenerator(self.experiment, self.settings)
            sequence_generator_dialog.update_sequence.connect(self.update_sequence)
            self.modules.append(sequence_generator_dialog)
        else:
            sequence_generator_dialog \
                = self.modules[[module.name for module in self.modules].index("Sequence Generator")]
        sequence_generator_dialog.show()

    def update_sequence(self, header, body, footer):
        self.experiment.update_sequence(header, body, footer)
        self.update_ui()

    def goto_movie(self):
        self.current_movie_num = self.ui.spinBoxMovie.value()
        self.current_movie = self.experiment.movies[self.current_movie_num]
        self.current_trajectory_num = 0
        self.current_trajectory = self.experiment.movies[self.current_movie_num].trajectories[0]
        self.update_ui()

    def toggle_movie(self):
        if self.ui.checkBoxMovie.isChecked():
            self.current_movie.set_inclusion(True)
            # print("movie included")
        else:
            self.current_movie.set_inclusion(False)
            # print("movie not included")

    def toggle_apply_to_experiment(self):
        if self.ui.checkBoxApplyToAllMovies.isChecked():
            self.apply_to_experiment = True
        else:
            self.apply_to_experiment = False

    def apply_correction(self):
        correction = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        correction[0] = self.ui.doubleSpinBoxDdBsl.value()
        correction[1] = self.ui.doubleSpinBoxAdBsl.value()
        correction[2] = self.ui.doubleSpinBoxAaBsl.value()
        correction[3] = self.ui.doubleSpinBoxReflection.value()
        correction[4] = self.ui.doubleSpinBoxDonorLeakage.value()
        correction[5] = self.ui.doubleSpinBoxDirectEx.value()
        correction[6] = self.ui.doubleSpinBoxDetection.value()
        correction[7] = self.ui.doubleSpinBoxNormalizedEx.value()
        if self.apply_to_experiment:
            for movie in self.experiment.movies.values():
                for trajectory in movie.trajectories.values():
                    trajectory.update_correction(correction)
        else:
            for trajectory in self.current_movie.trajectories.values():
                trajectory.update_correction(correction)
        self.update_ui()

    def reset_correction(self):
        correction = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        if self.apply_to_experiment:
            for movie in self.experiment.movies.values():
                for trajectory in movie.trajectories.values():
                    trajectory.update_correction(correction)
        else:
            for trajectory in self.current_movie.trajectories.values():
                trajectory.update_correction(correction)
        self.update_ui()

    def goto_trajectory(self):
        self.current_trajectory_num = self.ui.spinBoxTrajectory.value()
        self.current_trajectory = \
            self.experiment.movies[self.current_movie_num].trajectories[self.current_trajectory_num]
        self.update_ui()

    def toggle_skip_trajectory(self):
        if self.ui.checkBoxSkipTrajectories.isChecked():
            self.skip_trajectory = True
        else:
            self.skip_trajectory = False

    def next_trajectory(self):
        m = self.current_movie_num
        if self.current_trajectory_num < self.experiment.movies[m].trajectory_num-1:
            self.current_trajectory_num += 1
            if self.skip_trajectory:
                n = self.current_trajectory_num
                while n < self.experiment.movies[m].trajectory_num-1 \
                        and not self.experiment.movies[m].trajectories[n].trajectory_included:
                    n += 1
                if self.experiment.movies[m].trajectories[n].trajectory_included:
                    self.current_trajectory_num = n
                else:
                    self.current_trajectory_num -= 1
            self.current_trajectory = self.experiment.movies[m].trajectories[self.current_trajectory_num]
            self.update_ui()

    def previous_trajectory(self):
        m = self.current_movie_num
        if self.current_trajectory_num > 0:
            self.current_trajectory_num -= 1
            if self.skip_trajectory:
                n = self.current_trajectory_num
                while n > 0 \
                        and not self.experiment.movies[m].trajectories[n].trajectory_included:
                    n -= 1
                if self.experiment.movies[m].trajectories[n].trajectory_included:
                    self.current_trajectory_num = n
                else:
                    self.current_trajectory_num += 1
            self.current_trajectory = self.experiment.movies[m].trajectories[self.current_trajectory_num]
            self.update_ui()

    def toggle_trajectory(self):
        if self.ui.checkBoxTrajectory.isChecked():
            self.current_trajectory.set_inclusion(True)
            # print("trajectory included")
        else:
            self.current_trajectory.set_inclusion(False)
            # print("trajectory not included")
        self.update_ui()

    def set_category(self):
        self.current_trajectory.set_category(self.ui.spinBoxCategory.value())
        self.update_ui()

    def update_trajectory_data_points(self, event):
        # print(self.current_trajectory.data_points)
        data_point = (event.xdata, event.ydata, 0 if event.button == 1 else 1, \
                      0 if event.y > self.ui.plot.height()/2 else 1)
        self.current_trajectory.update_trajectory_data_points([data_point])

    def update_self_data_points(self, event):
        # print(self.data_points)
        data_point = event.xdata
        # an empty list is converted to None
        if self.data_points is None:
            self.data_points = [data_point]
        else:
            self.data_points.append(data_point)

    def click_data_points(self):
        if self.ui.toolButtonClickPoints.isChecked():
            self.ui.comboBoxFunctions.setEnabled(False)
            if self.ui.comboBoxFunctions.currentIndex() == 0:
                self.mid = self.canvas.mpl_connect('button_press_event', self.update_trajectory_data_points)
            elif self.ui.comboBoxFunctions.currentIndex() in [1, 2]:
                self.mid = self.canvas.mpl_connect('button_press_event', self.update_self_data_points)
        else:
            self.ui.comboBoxFunctions.setEnabled(True)
            self.canvas.mpl_disconnect(self.mid)
            if self.data_points is not None:
                if self.ui.comboBoxFunctions.currentIndex() == 1:
                    for i in range(1, len(self.data_points), 2):
                        start = self.data_points[i-1]
                        end = self.data_points[i]
                        self.current_trajectory.trim_trajectory(start, end)
                elif self.ui.comboBoxFunctions.currentIndex() == 2:
                    for i in range(1, min(len(self.data_points), 2), 2):
                        start = self.data_points[i - 1]
                        end = self.data_points[i]
                        self.current_trajectory.calculate_x_correlation(start, end)
                self.data_points = None
            self.update_ui()

    def update_time_offset(self):
        time_offset = self.ui.doubleSpinBoxTimeOffset.value()
        self.current_trajectory.update_time_offset(time_offset-self.current_trajectory.time_offset)
        self.update_ui()

    def reset_trajectory(self):
        time_offset = 0.0
        self.current_trajectory.x_correlation = np.array([])
        self.current_trajectory.update_time_offset(time_offset-self.current_trajectory.time_offset)
        self.current_trajectory.trim_trajectory(0, 0, True)
        self.data_points = None
        self.update_ui()

    def reset_movie(self):
        time_offset = 0.0
        for trajectory in self.current_movie.trajectories.values():
            trajectory.x_correlation = np.array([])
            trajectory.update_time_offset(time_offset-trajectory.time_offset)
            trajectory.trim_trajectory(0, 0, True)
        self.data_points = None
        self.update_ui()

    def save_trajectory(self):
        # print("save movie "+str(self.current_movie_num)+" trajectory "+str(self.current_trajectory_num))
        self.current_trajectory.save_trajectory()

    def clear_trajectory(self):
        self.current_trajectory.update_trajectory_data_points([], True)
        self.update_ui()

    def clear_movie(self):
        for trajectory in self.current_movie.trajectories.values():
            trajectory.update_trajectory_data_points([], True)
        self.update_ui()

    def save_data_points(self):
        print("save selected data points")

if __name__ == "__main__":
    # execute only if run as a script
    # necessary to import fret_tools_main for jupyter notebook
    import fret_tools_main

    app = QtWidgets.QApplication(sys.argv)

    # this keeps Qt from closing everything if a message box is displayed
    # before the main window is shown
    # app.setQuitOnLastWindowClosed(False)

    # splash screen
    pixmap = QtGui.QPixmap("splash.png")
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()
    app.processEvents()

    # set up the main window
    window = Window()

    splash.hide()
    window.show()

    app.exec_()


