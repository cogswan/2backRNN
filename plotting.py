"""
Functions for plotting stimulus
representation
"""

import numpy as np
import matplotlib.pyplot as plt

import pumi


def setup_fig(plots, size=4):
    return plt.subplots(1, plots, figsize=(plots * size, size), sharex=True, sharey=True)


def scatter(data_2d, ax, marker=None, label=None, center=True, symbol = True, line = False):
    
    plot_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    plot_markers = ['d', '^', 'X','P', 's', '*']

    if data_2d.shape[0] == 2:
        x, y = data_2d[0], data_2d[1]
        n = data_2d.shape[1]
    else:
        x, y = data_2d[:, 0], data_2d[:, 1]
        n = data_2d.shape[0]

    # ax.scatter(x, y, s=65, c=plot_colors,  marker=marker,label=label)
    for i in range(6):
        if symbol is True:
            ax.scatter(x[i], y[i], s=65, c='k',  marker=plot_markers[i], label=label)   
        else:
            ax.scatter(x[i], y[i], s=65, c=plot_colors[i],  marker='o', label=label)
    
    #connect data points with lines following stimulus order if EEG/60D RNN
    if line is True:
        x0 = np.append(x,x[0])
        y0 = np.append(y,y[0])
        ax.plot(x0,y0,color = '#969696')
    # ax.legend()
    
    if center:
        center_axes(ax)


def center_axes(ax, noticks=False):

    # Get rid of box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Center axes
    for s in ['left', 'bottom']:
        ax.spines[s].set_position(('data', 0))
        ax.spines[s].set_zorder(-1)

    # Get rid of axis ticks
    if noticks:
        ax.set_xticks([])
        ax.set_yticks([])

    # Match axes scales
    ax.set_aspect('equal')
