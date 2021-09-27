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

import ft_library.fitting_tools as fitting_tools

class Trajectory:
    def __init__(self, settings, tn, **kwargs):
        self.settings = settings
        self.name = tn
        self.directory = self.settings['directory']
        self.fsl = self.settings['front_sample_length']
        self.bsl = self.settings['back_sample_length']
        self.time_unit = self.settings['time_unit']
        self.trajectory_included = False
        self.category = 1

        # self.raw_data is typically read from the arguments
        # however, when the Trajectory class is called independently
        # self.raw_data can be read from .dat files
        if 'raw_data' in list(kwargs.keys()):
            self.raw_data = kwargs['raw_data']
        else:
            self.raw_data = self.read_data()
        assert self.raw_data.shape[1] == 3
        assert self.time_unit == self.raw_data[1, 0] - self.raw_data[0, 0]
        self.length = self.raw_data.shape[0]

        # corrected_data is a copy of raw_data. raw_data should almost always be treated as immutable
        self.corrected_data = np.copy(self.raw_data)
        self.corrected_data = np.concatenate((self.corrected_data, np.ones((self.length, 1))), axis=1)

        self.illumination = np.zeros((self.length, 3))
        self.efret = np.zeros((self.length, ))
        self.smoothed_efret = np.zeros((self.length, ))
        self.update_efret()

        self.time_offset = 0.0
        self.data_points = None
        self.correction = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        self.bins = np.arange(-0.5, 1.525, 0.025)
        self.fret_histogram = np.array([])
        self.make_fret_histogram()
        self.x_correlation = np.array([])

    # not used by fret_tools_main.py
    # use this function to read existing .dat files
    def read_data(self):
        path = os.path.join(self.directory, self.name)
        with open(path, "r") as file:
            raw_data = np.loadtxt(file)
        return raw_data

    def update_sequence(self, header, body, footer):
        # print("update sequence")
        header = header.T
        body = body.T
        footer = footer.T

        assert np.shape(header)[1] == np.shape(self.illumination)[1]
        assert np.shape(body)[1] == np.shape(self.illumination)[1]
        assert np.shape(footer)[1] == np.shape(self.illumination)[1]

        self.illumination[0:np.shape(header)[0], :] = header
        self.illumination[self.length-np.shape(footer)[0]:, :] = footer
        length = self.length - np.shape(header)[0] - np.shape(footer)[0]
        self.illumination[np.shape(header)[0]:self.length-np.shape(footer)[0], :] \
            = np.resize(body, (length, np.shape(self.illumination)[1]))
        self.update_efret()
        self.make_fret_histogram()

    def update_time_offset(self, time_offset=0.0):
        self.time_offset = self.time_offset + time_offset
        self.apply_time_offset()
        self.apply_time_offset_data_points(time_offset)

    def apply_time_offset(self):
        self.corrected_data[:, 0] = self.raw_data[:, 0] - self.time_offset

    def apply_time_offset_data_points(self, time_offset=0.0):
        if self.data_points is not None:
            for n in np.arange(0, len(self.data_points)):
                self.data_points[n] = (self.data_points[n][0] - time_offset, self.data_points[n][1], \
                                       self.data_points[n][2], self.data_points[n][3])

    def set_inclusion(self, inclusion):
        self.trajectory_included = inclusion

    def set_category(self, category):
        self.category = category

    def update_correction(self, correction=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])):
        assert correction.size == 8
        self.correction = correction
        self.apply_correction()

    def apply_correction(self):
        ddbsl = self.correction[0]
        adbsl = self.correction[1]
        aabsl = self.correction[2]
        reflection = self.correction[3]
        leakage = self.correction[4]
        directex = self.correction[5]
        detection = self.correction[6]
        normalex = self.correction[7]

        index = self.illumination[:, 1].astype(bool)
        donor = np.copy(self.raw_data[~index, 1])
        acceptor = np.copy(self.raw_data[~index, 2])
        acceptora = np.copy(self.raw_data[index, 2])

        donor = donor - ddbsl
        acceptor = acceptor - adbsl
        acceptora = acceptora - aabsl
        tempaa = np.mean(acceptora[1:self.fsl - 1])
        donor = donor - acceptor * reflection
        acceptor = acceptor + acceptor * reflection
        acceptora = acceptora + acceptora * reflection
        acceptor = acceptor - leakage * donor - directex * tempaa
        donor = donor * detection
        acceptora = np.divide(acceptora, normalex)

        self.corrected_data[~index, 1] = donor
        self.corrected_data[~index, 2] = acceptor
        self.corrected_data[index, 2] = acceptora
        del donor
        del acceptor
        del acceptora
        self.update_efret()
        self.make_fret_histogram()

    def update_efret(self):
        index = self.illumination[:, 1].astype(bool)
        efret = np.divide(self.corrected_data[~index, 2], self.corrected_data[~index, 1]+self.corrected_data[~index, 2])
        self.efret[~index] = efret
        self.efret[index] = -1.0
        self.smooth_efret()

    def smooth_efret(self, window=3):
        index = self.illumination[:, 1].astype(bool)
        smoothed_efret = signal.medfilt(self.efret[~index], window)
        self.smoothed_efret[~index] = smoothed_efret
        self.smoothed_efret[index] = -1.0

    # TODO: take into account the fourth column of corrected_data
    def contribute_to_pre_plots_point(self):
        index = self.illumination[:, 1].astype(bool)
        donor = np.copy(self.raw_data[~index, 1])
        acceptor = np.copy(self.raw_data[~index, 2])
        donora = np.copy(self.raw_data[index, 1])
        acceptora = np.copy(self.raw_data[index, 2])

        tempdd = donor[-self.bsl + 1:-1]
        tempad = acceptor[-self.bsl + 1:-1]
        tempda = donora[-self.bsl + 1:-1]
        tempaa = acceptora[-self.bsl + 1:-1]

        del donor
        del acceptor
        del donora
        del acceptora

        return tempdd, tempad, tempda, tempaa

    # TODO: take into account the fourth column of corrected_data
    def contribute_to_e_s_plot_average(self, start=1, length=8):
        index = self.illumination[:, 1].astype(bool)
        donor = np.copy(self.raw_data[~index, 1])
        acceptor = np.copy(self.raw_data[~index, 2])
        acceptora = np.copy(self.raw_data[index, 2])

        tempdd = np.mean(donor[start:start + length])
        tempad = np.mean(acceptor[start:start + length])
        tempaa = np.mean(acceptora[start:start + length])
        if math.isnan(tempdd) or math.isnan(tempad):
            tempe = -10.0
            temps = -10.0
        else:
            tempe = np.divide(tempad, tempdd + tempad)
            temps = np.divide(tempdd + tempad, tempdd + tempad + tempaa)

        del donor
        del acceptor
        del acceptora

        return tempaa, tempdd + tempad, tempe, temps

    # TODO: take into account the fourth column of corrected_data
    def contribute_to_fret_histogram_point(self, start=1, length=8):
        index = self.illumination[:, 1].astype(bool)
        smoothed_efret = self.smoothed_efret[~index]

        return smoothed_efret[start:start + length]

    def plot_trajectory(self, **kwargs):
        if 'axes' in kwargs.keys():
            axes = kwargs['axes']
        else:
            _, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 5))

        ##Set the position of the figure
        axes[0].set_position([0.11, 0.55, 0.85, 0.35])
        axes[1].set_position([0.11, 0.15, 0.85, 0.35])

        for ax in axes:
            ax.clear()
            ##Set spine width
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ##Remove the top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        index = (self.efret < np.inf) * (self.efret > -np.inf)
        xdata = self.corrected_data[index, 0]
        ydata = self.corrected_data[index, 1:]
        efret = self.efret[index]
        illumination = self.illumination[index, :]
        index = illumination[:, 1].astype(bool)

        ##Plot data points
        axes[0].plot(xdata[~index], ydata[~index, 0], linewidth=1, color='g', zorder=1)
        axes[0].plot(xdata[~index], ydata[~index, 1], linewidth=1, color=(1.0, 0.3, 0.3), zorder=1)
        axes[1].plot(xdata[~index], efret[~index], linewidth=1, color='k', zorder=1)

        axes[0].plot(xdata[index], ydata[index, 1], linewidth=1, color='m', zorder=2)

        axes[0].vlines(xdata[~ydata[:, -1].astype(bool)], -10000.0, 10000.0, colors='w', zorder=3)
        axes[1].vlines(xdata[~ydata[:, -1].astype(bool)], -10.0, 10.0, colors='w', zorder=3)

        if self.data_points is not None:
            for data_point in self.data_points:
                axes[data_point[3]].scatter(data_point[0], data_point[1], s=3, c='b' if data_point[2] else 'c')

        ##Set axis ranges
        maxInt = np.max([np.max(ydata[:, 0]), np.max(ydata[:, 1])])
        axes[0].axis([-1.0, np.max([xdata[-1], 10.0]), -maxInt * 0.1, maxInt * 1.1])
        axes[1].axis([-1.0, np.max([xdata[-1], 10.0]), -0.2, 1.2])

        ##Set tick ranges
        if xdata[-1] < 50.0:
            step = 5
        elif xdata[-1] < 100.0:
            step = 10
        elif xdata[-1] < 250.0:
            step = 25
        elif xdata[-1] < 1000.0:
            step = 50
        else:
            step = 100
        for ax in axes:
            ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, np.max([xdata[-1], 10.0]) * 1.01, step)))
            # ax.xaxis.set_major_formatter(ticker.NullFormatter())
        axes[1].yaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1.2, 0.4)))

        ##Set tick parameters
        for ax in axes:
            ax.xaxis.set_tick_params(which='major', length=3, width=1, labelsize=8)
            ax.yaxis.set_tick_params(which='major', length=3, width=1, labelsize=8)

        ##Add axis labels and set their font sizes and positions
        axes[1].xaxis.set_label_text('$Time\ (s)$', size=12)
        axes[0].yaxis.set_label_text('$Intensity$', size=12)
        axes[1].yaxis.set_label_text('$E_{FRET}$', size=12)
        axes[1].xaxis.set_label_coords(0.5, -0.18)
        axes[0].yaxis.set_label_coords(-0.08, 0.5)
        axes[1].yaxis.set_label_coords(-0.08, 0.5)

        plt.show()

    def save_trajectory(self):
        with open(self.directory + self.name, "w") as file:
            np.savetxt(file, self.corrected_data, fmt='%.7f')

    def update_trajectory_data_points(self, data_points, reset=False):
        if reset:
            self.data_points = None
        else:
            if self.data_points is None:
                self.data_points = data_points
            else:
                for data_point in data_points:
                    self.data_points.append(data_point)

    def trim_trajectory(self, start, end, reset=False):
        if reset:
            self.corrected_data[:, -1] = 1
        else:
            if start is not None and end is not None:
                start = np.searchsorted(self.corrected_data[:, 0], start)
                end = np.searchsorted(self.corrected_data[:, 0], end)
                self.corrected_data[start:end, -1] = 0
            else:
                print("please click within the boundaries")
        self.make_fret_histogram()

    # TODO: try to make correlation work with unevenly spaced time series
    def calculate_x_correlation(self, start, end):
        start = np.searchsorted(self.corrected_data[:, 0], start)
        end = np.searchsorted(self.corrected_data[:, 0], end)
        if start < end:
            mean_g = np.mean(self.corrected_data[start:end, 1])
            mean_r = np.mean(self.corrected_data[start:end, 2])
            x_correlation \
                = np.correlate(self.corrected_data[start:end, 1]-mean_g, self.corrected_data[start:end, 2]-mean_r, \
                               "full")
            x_correlation \
                = np.copy(x_correlation[int(np.floor(x_correlation.size / 2)):] / (mean_g * mean_r))
            self.x_correlation = x_correlation / np.arange(x_correlation.size, 0, -1)
            # plt.plot(np.arange(0, self.x_correlation.size), self.x_correlation)
            # plt.show()

    def plot_x_correlation(self, **kwargs):
        if 'axis' in kwargs.keys():
            axis = kwargs['axis']
        else:
            _, axis = plt.subplots(1, 1, figsize=(3, 5))

        ##Set the position of the figure
        axis.set_position([0.16, 0.15, 0.82, 0.83])

        ##Set spine width
        axis.spines['bottom'].set_linewidth(1.5)
        axis.spines['left'].set_linewidth(1.5)

        ##Remove the top and right spines
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

        ##Plot data points
        axis.plot(self.time_unit * np.arange(0, self.x_correlation.size), self.x_correlation, linewidth=0.3)

        ##Set axis ranges
        # maxCount = np.max(self.x_correlation)
        # axis.axis([0, self.time_unit * cross_correlation.size, 0, maxCount * 1.2])

        ##Set tick parameters
        axis.xaxis.set_tick_params(which='major', length=2, width=1, labelsize=3)
        axis.yaxis.set_tick_params(which='major', length=2, width=1, labelsize=3)

        ##Add axis labels and set their font sizes and positions
        axis.xaxis.set_label_text('$Time\ (s)$', size=5)
        axis.yaxis.set_label_text('$Normalized\ Cross-Correlation$', size=5)
        axis.xaxis.set_label_coords(0.5, -0.10)
        axis.yaxis.set_label_coords(-0.12, 0.5)

        plt.show()

    def make_fret_histogram(self):
        index = np.multiply(self.corrected_data[:, -1].astype(bool), ~self.illumination[:, 1].astype(bool))
        self.fret_histogram, _ = np.histogram(self.smoothed_efret[index], bins=self.bins, density=True)

    def plot_fret_histogram(self, **kwargs):
        if 'axis' in kwargs.keys():
            axis = kwargs['axis']
        else:
            _, axis = plt.subplots(1, 1, figsize=(3, 5))

        ##Set the position of the figure
        axis.set_position([0.16, 0.15, 0.82, 0.83])

        ##Set spine width
        axis.spines['bottom'].set_linewidth(1.5)
        axis.spines['left'].set_linewidth(1.5)

        ##Remove the top and right spines
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

        ##Plot data points
        axis.bar(self.bins[0:-1], self.fret_histogram, width=np.diff(self.bins)[0], align='edge')

        ##Set axis ranges
        maxCount = np.max(self.fret_histogram)
        try:
            axis.axis([-0.2, 1.2, 0, maxCount * 1.2])
        except ValueError:
            axis.axis([-0.2, 1.2, 0, 10.0 * 1.2])

        ##Set tick ranges
        axis.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1.2, 0.2)))

        ##Set tick parameters
        axis.xaxis.set_tick_params(which='major', length=2, width=1, labelsize=3)
        axis.yaxis.set_tick_params(which='major', length=2, width=1, labelsize=3)

        ##Add axis labels and set their font sizes and positions
        axis.xaxis.set_label_text('$E_{FRET}$', size=5)
        axis.yaxis.set_label_text('$Probability$', size=5)
        axis.xaxis.set_label_coords(0.5, -0.10)
        axis.yaxis.set_label_coords(-0.12, 0.5)

        plt.show()


