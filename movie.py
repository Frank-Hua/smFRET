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
import ft_library.fitting_tools as fitting_tools

class Movie:
    def __init__(self, settings, mn):
        self.settings = settings
        self.name = mn
        self.directory = self.settings['directory']
        self.time_unit = self.settings['time_unit']
        self.trajectories = {}
        self.trajectory_num = 0
        self.movie_included = True

        self.generate_trajectories()

    def generate_trajectories(self):
        with open(os.path.join(self.directory, self.name), "rb") as file:
            mlength = np.fromfile(file, dtype=np.int32, count=1)
            mlength = mlength[0]
            ntrace = np.fromfile(file, dtype=np.int16, count=1)
            ntrace = ntrace[0]
            raw_data = np.fromfile(file, dtype=np.int16, count=ntrace * mlength)
        raw_data.resize((mlength, ntrace))

        # Assume the number of channels is 2, e.g. Cy3 and Cy5
        nmolecule = int(ntrace / 2)
        xdata = self.time_unit * np.arange(0, mlength)
        xdata.resize(mlength, 1)
        for n in np.arange(0, nmolecule):
            trajectory_name = self.name[:-7] + '_tr' + str(n) + '.dat'
            trajectory_data = np.concatenate((xdata, np.float64(raw_data[:, n*2:n*2+2])), axis=1)
            self.trajectories[n] = Trajectory(self.settings, trajectory_name, raw_data=trajectory_data)
        self.trajectory_num = nmolecule

    def set_inclusion(self, inclusion):
        self.movie_included = inclusion


