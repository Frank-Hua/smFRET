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
import ft_library.fitting_tools as fitting_tools

class Experiment:
    def __init__(self, settings):
        self.settings = settings
        self.name = self.settings['experiment_name'] + '.pkl'
        self.directory = self.settings['directory']
        self.fsl = self.settings['front_sample_length']
        self.bsl = self.settings['back_sample_length']
        self.movies = {}
        self.movie_num = 0
        self.open_movies()

        self.header_sequence = np.zeros((3, 0), dtype=np.int8)
        self.body_sequence = np.resize(np.array([1, 0, 0], dtype=np.int8), (3, 1))
        self.footer_sequence = np.zeros((3, 0), dtype=np.int8)
        self.initialize_sequence()
        self.update_sequence(self.header_sequence, self.body_sequence, self.footer_sequence)
        self.bins = self.movies[0].trajectories[0].bins

        self.make_pickle()

    def open_movies(self):
        for fn in os.listdir(self.directory):
            if '.traces' in fn and os.path.isfile(os.path.join(self.directory, fn)):
                movie_name = fn
                self.movies[self.movie_num] = Movie(self.settings, movie_name)
                self.movie_num += 1

    def make_pickle(self):
        with open(self.directory + self.name, "wb") as file:
            pickle.dump(self, file)

    def initialize_sequence(self):
        seq = np.array([0, 1, 0])
        for _ in np.arange(0, self.fsl):
            self.header_sequence = np.concatenate((self.header_sequence, np.resize(seq, (3, 1))), axis=1)
        for _ in np.arange(0, self.bsl):
            self.footer_sequence = np.concatenate((self.footer_sequence, np.resize(seq, (3, 1))), axis=1)

    def update_sequence(self, header, body, footer):
        self.header_sequence = header
        self.body_sequence = body
        self.footer_sequence = footer
        for movie in self.movies.values():
            for trajectory in movie.trajectories.values():
                trajectory.update_sequence(header, body, footer)

    def make_intensity_histogram(self):
        intensity_a = []
        intensity_d = []
        for movie in self.movies.values():
            if movie.movie_included:
                for trajectory in movie.trajectories.values():
                    tempa, tempd, _, _ = trajectory.contribute_to_e_s_plot_average(length=self.fsl-2)
                    intensity_a.append(tempa)
                    intensity_d.append(tempd)
        intensity_a = np.array(intensity_a)
        intensity_d = np.array(intensity_d)
        maxIntensity = np.max(np.append(intensity_a, intensity_d))
        bins = np.arange(-maxIntensity*0.25, maxIntensity*1.25, maxIntensity/60.0)
        intensity_figure, intensity_axis = plt.subplots(1, 1)
        intensity_axis.set_ylim(-1, 0)
        counts_a, _ = np.histogram(intensity_a, bins=bins)
        counts_d, _ = np.histogram(intensity_d, bins=bins)
        intensity_figure.canvas.manager.set_window_title("Intensity Histogram")
        self.plot_intensity_histogram(bins[0:-1], counts_a, (1.0, 0.3, 0.3), 1, intensity_axis)
        self.plot_intensity_histogram(bins[0:-1], counts_d, 'g', 2, intensity_axis)
        intensity_figure.legend(['Acceptor labeling', 'Donor labeling'], fontsize='large', frameon=False)

    def plot_intensity_histogram(self, bins, counts, color, zorder, axis):
        ##Set the position of the figure
        axis.set_position([0.20, 0.15, 0.78, 0.83])

        ##Set spine width
        axis.spines['bottom'].set_linewidth(1.5)
        axis.spines['left'].set_linewidth(1.5)

        ##Remove the top and right spines
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

        ##Plot data points
        axis.step(bins, counts, color=color, linewidth=1.5, zorder=zorder)

        ##Set axis ranges
        _, ymax = axis.get_ylim()
        maxCount = np.max(np.append(counts, ymax))
        axis.axis([bins[0], bins[-1]*1.1, 0, maxCount * 1.2])

        ##Set tick ranges
        axis.xaxis.set_major_locator(ticker.FixedLocator(bins[::15]))

        ##Set tick parameters
        axis.xaxis.set_tick_params(which='major', length=5, width=1.5, labelsize=15)
        axis.yaxis.set_tick_params(which='major', length=5, width=1.5, labelsize=15)

        ##Add axis labels and set their font sizes and positions
        axis.xaxis.set_label_text('$Intensity$', size=15)
        axis.yaxis.set_label_text('$Probability$', size=15)
        axis.xaxis.set_label_coords(0.5, -0.10)
        axis.yaxis.set_label_coords(-0.17, 0.5)

        plt.show()

    def make_pre_plots(self, cutoff=0):
        dd = []
        ad = []
        da = []
        aa = []
        for movie in self.movies.values():
            if movie.movie_included:
                for trajectory in movie.trajectories.values():
                    tempdd, tempad, tempda, tempaa = trajectory.contribute_to_pre_plots_point()
                    dd.append(tempdd)
                    ad.append(tempad)
                    da.append(tempda)
                    aa.append(tempaa)
        dd = np.array(dd)
        dd = np.resize(dd, (np.size(dd), 1))
        ad = np.array(ad)
        ad = np.resize(ad, (np.size(ad), 1))
        da = np.array(da)
        da = np.resize(da, (np.size(da), 1))
        aa = np.array(aa)
        aa = np.resize(aa, (np.size(aa), 1))

        maxIntensity = np.max(aa)
        bins = np.arange(-maxIntensity * 0.25, maxIntensity * 1.25, maxIntensity / 60.0)
        fig, axis = plt.subplots(1, 1)
        axis.hist(aa, bins=bins)
        fig.canvas.manager.set_window_title("Intensity Histogram")
        fig.legend(['Acceptor labeling'], fontsize='large', frameon=False)
        axis.xaxis.set_label_text('$Intensity$', size=15)
        axis.yaxis.set_label_text('$Counts$', size=15)

        index = aa < cutoff
        dd = dd[index]
        ad = ad[index]
        da = da[np.invert(index)]
        aa = aa[np.invert(index)]
        (popt, pcov) = fitting_tools.linearFitter(aa, da, 0)
        print(f"linear fitting parameters: slope-{popt[0]:.3f} y-intercept-{popt[1]:.3f}")

        bins = np.arange(-100, maxIntensity * 1.25, 5)
        fig, axes = plt.subplots(2, 1)
        axes[0].hist(dd, bins=bins)
        axes[1].hist(ad, bins=bins)
        fig.canvas.manager.set_window_title("Intensity Histograms")
        axes[0].legend(['Donor baseline'], fontsize='large', frameon=False)
        axes[1].legend(['Acceptor baseline'], fontsize='large', frameon=False)
        axes[0].yaxis.set_label_text('$Counts$', size=15)
        axes[1].xaxis.set_label_text('$Intensity$', size=15)
        axes[1].yaxis.set_label_text('$Counts$', size=15)
        plt.show()

        fig, axis = plt.subplots(1, 1)
        axis.scatter(aa, da, s=10, marker='x', linewidths=0.5)
        axis.plot(np.arange(0, np.max(aa)), fitting_tools.linear(np.arange(0, np.max(aa)), *popt), \
                  color='k', linewidth=2)
        fig.canvas.manager.set_window_title("Intensity Scatter Plot-Reflection")
        axis.xaxis.set_label_text('$Acceptor\ intensity$', size=15)
        axis.yaxis.set_label_text('$Donor\ intensity$', size=15)
        plt.show()

    def make_e_s_2dhistogram(self):
        e = []
        s = []
        for movie in self.movies.values():
            if movie.movie_included:
                for trajectory in movie.trajectories.values():
                    tempa, tempd, tempe, temps = trajectory.contribute_to_e_s_plot_average(length=self.fsl-2)
                    e.append(tempe)
                    s.append(temps)
        e = np.array(e)
        s = np.array(s)
        # plt.hist2d(e, s, bins=80)
        xedges = np.linspace(-0.50, 1.50, 81)
        yedges = np.linspace(-0.50, 1.50, 81)
        e_s_figure, e_s_axis = plt.subplots(1, 1)
        (counts, xedges, yedges) = np.histogram2d(e, s, bins=[xedges, yedges])
        e_s_figure.canvas.manager.set_window_title("E-S 2D Histogram")
        self.plot_e_s_2dhistogram(xedges, yedges, counts, e_s_axis)

    def plot_e_s_2dhistogram(self, xedges, yedges, counts, axis):
        ##Set the position of the figure
        axis.set_position([0.15, 0.15, 0.83, 0.83])

        ##Set spine width
        axis.spines['top'].set_linewidth(1.5)
        axis.spines['bottom'].set_linewidth(1.5)
        axis.spines['right'].set_linewidth(1.5)
        axis.spines['left'].set_linewidth(1.5)

        peak = np.max(counts.T)
        axis.contour(xedges[1:]-0.0125, yedges[1:]-0.0125, counts.T,
                     cmap=plt.get_cmap('Blues'), norm=colors.Normalize(vmin=0, vmax=0.16 * peak),
                     levels=[0.02 * peak, 0.04 * peak, 0.08 * peak, 0.12 * peak, 0.16 * peak], origin='lower')

        ##Set axis ranges
        axis.axis([-0.2, 1.2, -0.2, 1.2])

        ##Set tick ranges
        axis.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1.2, 0.2)))
        axis.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1.2, 0.2)))

        ##Set tick parameters
        axis.xaxis.set_tick_params(which='major', length=5, width=1.5, labelsize=15)
        axis.yaxis.set_tick_params(which='major', length=5, width=1.5, labelsize=15)

        ##Add axis labels and set their font sizes and positions
        axis.xaxis.set_label_text('$E_{FRET}$', size=15)
        axis.yaxis.set_label_text('$Stoichiometry$', size=15)
        axis.xaxis.set_label_coords(0.5, -0.10)
        axis.yaxis.set_label_coords(-0.10, 0.5)

        plt.show()

    def make_fret_histogram(self):
        fret_data = []
        for movie in self.movies.values():
            if movie.movie_included:
                for trajectory in movie.trajectories.values():
                    fret_data += trajectory.contribute_to_fret_histogram_point(length=self.fsl-2).tolist()
        fret_data = np.array(fret_data)
        # plt.hist(fret_data, bins)
        fret_figure, fret_axis = plt.subplots(1, 1)
        counts, _ = np.histogram(fret_data, self.bins)
        fret_figure.canvas.manager.set_window_title("Overall FRET Histogram")
        self.plot_fret_histogram(self.bins[0:-1], counts, fret_axis)

    def plot_fret_histogram(self, bins, counts, axis):
        ##Set the position of the figure
        axis.set_position([0.20, 0.15, 0.78, 0.83])

        ##Set spine width
        axis.spines['bottom'].set_linewidth(1.5)
        axis.spines['left'].set_linewidth(1.5)

        ##Remove the top and right spines
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

        ##Plot data points
        axis.step(bins, counts, color='k', linewidth=1.5)

        ##Set axis ranges
        maxCount = np.max(counts)+0.1
        axis.axis([-0.2, 1.2, 0, maxCount * 1.2])

        ##Set tick ranges
        axis.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1.2, 0.2)))

        ##Set tick parameters
        axis.xaxis.set_tick_params(which='major', length=5, width=1.5, labelsize=15)
        axis.yaxis.set_tick_params(which='major', length=5, width=1.5, labelsize=15)

        ##Add axis labels and set their font sizes and positions
        axis.xaxis.set_label_text('$E_{FRET}$', size=15)
        axis.yaxis.set_label_text('$Probability$', size=15)
        axis.xaxis.set_label_coords(0.5, -0.10)
        axis.yaxis.set_label_coords(-0.17, 0.5)

        plt.show()

    # TODO: calculate FRET histograms by category
    def calculate_fret_histogram(self):
        counts = np.zeros_like(self.bins[0:-1])
        for movie in self.movies.values():
            if movie.movie_included:
                for trajectory in movie.trajectories.values():
                    if trajectory.trajectory_included:
                        counts += trajectory.fret_histogram
        fret_figure, fret_axis = plt.subplots(1, 1)
        fret_figure.canvas.manager.set_window_title("FRET Histogram By Category")
        self.plot_fret_histogram(self.bins[0:-1], counts, fret_axis)

    def calculate_cross_correlation(self):
        cross_correlation = np.zeros((100000, 2))
        for movie in self.movies.values():
            if movie.movie_included:
                for trajectory in movie.trajectories.values():
                    if trajectory.trajectory_included:
                        while trajectory.category > cross_correlation.shape[1] / 2:
                            cross_correlation \
                                = np.concatenate((cross_correlation, np.zeros((100000, 2))), axis=1)
                        length = trajectory.x_correlation.size
                        x_correlation = np.resize(trajectory.x_correlation, (length, 1))
                        cross_correlation[:length, 2*trajectory.category-2:2*trajectory.category] \
                            += np.concatenate((x_correlation, np.ones((length, 1))), axis=1)
        for n in np.arange(0, cross_correlation.shape[1], 2):
            cross_correlation_figure, cross_correlation_axis = plt.subplots(1, 1)
            cross_correlation[:, n] = cross_correlation[:, n] / cross_correlation[:, n+1]
            cross_correlation_figure.canvas.manager.set_window_title\
                (f"Cross-Correlation of Trajectory Category {int(n/2)+1}")
            self.plot_cross_correlation(cross_correlation[:, n], cross_correlation_axis)

    def plot_cross_correlation(self, cross_correlation, axis):
        ##Set the position of the figure
        axis.set_position([0.20, 0.15, 0.78, 0.83])

        ##Set spine width
        axis.spines['bottom'].set_linewidth(1.5)
        axis.spines['left'].set_linewidth(1.5)

        ##Remove the top and right spines
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

        ##Plot data points
        axis.plot(self.settings['time_unit'] * np.arange(0, cross_correlation.size), cross_correlation, linewidth=1.5)

        ##Set axis ranges
        # maxCount = np.max(cross_correlation)
        # axis.axis([0, self.settings['time_unit'] * cross_correlation.size, 0, maxCount * 1.2])

        ##Set tick parameters
        axis.xaxis.set_tick_params(which='major', length=5, width=1.5, labelsize=15)
        axis.yaxis.set_tick_params(which='major', length=5, width=1.5, labelsize=15)

        ##Add axis labels and set their font sizes and positions
        axis.xaxis.set_label_text('$Time\ (s)$', size=15)
        axis.yaxis.set_label_text('$Normalized\ Cross-Correlation$', size=15)
        axis.xaxis.set_label_coords(0.5, -0.10)
        axis.yaxis.set_label_coords(-0.17, 0.5)

        plt.show()


