"""Assorted plotting functions.

Jan. 2018
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_score_std(x_ax, score, title):

    plt.plot(x_ax, score.mean(0), color='steelblue')
    ax = plt.gca()
    ax.fill_between(x_ax,
                    score.mean(0) - np.std(score),
                    score.mean(0) + np.std(score),
                    alpha=.4, color='steelblue')
    plt.axvline(x=0., color='black')
    plt.ylabel('AUC')
    plt.xlim(x_ax[0], x_ax[-1])
    plt.xlabel('time')
    plt.title(title)
    plt.show()
