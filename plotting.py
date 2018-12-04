"""Assorted plotting functions.

AUTHOR: Britta U. Westner <britta.wstnr[at]gmail.com>
LICENCE: BSD 3-clause
"""
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img


def plot_score_std(x_ax, scores, title=None, colors=None, legend=None):

    if colors is None:
        colors = ['mediumseagreen', 'crimson', 'steelblue']
        if len(scores) > 3:
            raise ValueError("Please specify colors for plotting.")

    for ii, score in enumerate(scores):
        plt.plot(x_ax, score.mean(0), color=colors[ii])
        ax = plt.gca()
        ax.fill_between(x_ax,
                        score.mean(0) - np.std(score),
                        score.mean(0) + np.std(score),
                        alpha=.4, color=colors[ii])

    plt.axvline(x=0., color='black')
    plt.ylabel('AUC')
    plt.xlim(x_ax[0], x_ax[-1])
    plt.xlabel('time')
    plt.title(title)
    if legend is not None:
        plt.legend(legend)


def plot_source_act(stc, fwd, mri=None, threshold=None, thresh_ref=None,
                    title=None, timepoint=None, save_fig=False,
                    fig_fname=None, cmap=None, vmax=None, coords=None,
                    add_coords=False):
    """Plot source activity on volume.

    Plots source activity on subject's MRI.

    Parameters:
    -----------
    stc : dict
        MNE Python beamformer output
    fwd : forward operator
        MNE forward model
    mri : string | None
        Can be path to a specific subject's brain or None for not having
        any background image.
    threshold : float | 'auto' | None
        Threshold for plotting, if 'auto', nilearn's automatic threshold is
        used, if None, no thresholding is done.
    thresh_ref : string
        Reference for thresholding. Can be 'all' to use maximum across time and
        space or 'max_time' to use maximum time point or 'timepoint' to refer
        to the time point given in timepoint.
    title : string | None
        Title for the figure.
    timepoint : float | string
        Time point that should be plotted. Can be given as index (int) or can
        be 'max' to select the time point with maximal activity.
    save_fig : bool
        whether the figure should be saved
    fig_fname : string
        where to save the figure to
    cmap : None | string
        Matplotlib color map for plotting, passed to nilearn's plot_stat_map.
        Popular choices might be  "viridis" or "RdBu". From the nilearn doc:
        The colormap must be symmetric. If None, the default color map will be
        used."
    vmax : None | float
        Upper (and -lower) limit of the color bar.
    coords : None | list of tuples
        Coordinates to cut and/or plot a marker at (see add_coords).
    add_coords : bool
        If True, a marker will be displayed at the coordinates provided in
        coords.

    Returns
    -------
    nilearn figure.
    """
    img = stc.as_volume(fwd['src'], mri_resolution=True)

    if timepoint is 'max':
        vox, timepoint = np.unravel_index(stc.data.argmax(), stc.data.shape)

    if thresh_ref is 'all':
        threshold = np.max(stc.data) * threshold
    elif thresh_ref is 'max_time':
        if timepoint is not 'max':
            # in that case, maximum time point needs to be calculated now:
            _, m_tp = np.unravel_index(stc.data.argmax(), stc.data.shape)
        threshold = np.max(stc.data[:, m_tp]) * threshold
    elif thresh_ref is 'timepoint':
        threshold = np.max(stc.data[:, timepoint] * threshold)

    if save_fig is True:
        if fig_fname is None:
            raise ValueError("Please specify a file name to save figure to.")
        if add_coords is True:
            raise NotImplementedError("Cannot plot markers and save yet, "
                                      "sorry.")
    else:
        fig_fname = None

    display = plot_stat_map(index_img(img, timepoint), bg_img=mri,
                            threshold=threshold, title=title, cmap=cmap,
                            symmetric_cbar=True, vmax=vmax,
                            output_file=fig_fname, cut_coords=coords[0],
                            display_mode='ortho')

    if add_coords is True:
        if coords is None:
            raise ValueError("Please provide coords for adding a marker.")
        # add a marker
        colors = ['w', 'y', 'b']
        for coord, color in zip(coords, colors):
            display.add_markers([coord], marker_color=color, marker_size=50)


def plot_source_ts(stc, n_ts, abs=True, xlims=None, ylims=None, title=None,
                   save_fig=False, fig_fname=None):
    """Plot source time series.

    Plots the n maximal time series in source space data.

    Parameters:
    -----------
    stc : dict
        MNE-Python source estimate.
    n_ts : int
        Number of time series to plot.
    abs : bool
        Whether the n time series should be picked on max() or max(abs()).
    xlims : tuple | None
        x axis limits for figure.
    ylims : tuple | None
        y axis limits for figure.
    title : string | None
        Title for  the figure.
    save_fig : bool
        Whether figure should be saved to disk. Note that the figure will not
        be shown in this case (nilearn properties).
    fig_fname : str
        Path for saving figure if save_fig=True.

    Returns
    -------
    matplotlib figure
    """
    plt.figure()
    if abs:
        plt.plot(stc.times,
                 stc.data[np.argsort(np.max(np.abs(stc.data), axis=1))
                          [-n_ts:]].T)
    else:
        plt.plot(stc.times,
                 stc.data[np.argsort(np.max(stc.data, axis=1))[-n_ts:]].T)

    # figure axes and title
    plt.xlabel('Time [s]')
    plt.ylabel('LCMV value [a.u.]')
    if xlims is not None:
        plt.xlim(xlims)
    else:
        plt.xlim(stc.times.min(), stc.times.max())
    if ylims is not None:
        plt.ylim(ylims)
    plt.title(title)
    plt.show()

    if save_fig is True:
        if fig_fname is None:
            raise ValueError("Please give a figure name to save to.")
        plt.savefig(fig_fname, bbox_inches='tight')


def plot_covariance(cov, title=None, colorbar=True, show_fig=True,
                    save_fig=False, fig_fname=None):
    """Plot covariance matrix.

    Plots covariance matrices.

    Parameters:
    -----------
    cov : covariance matrix
        MNE-Python covaraince matrix instance.
    title : str
        Title for plot.
    colorbar : bool
        Should color bar be added? Defaults to True.
    show_fig :  bool
        Whether figure should be displayed.
    save_fig : bool
        Whether figure should be saved to disk. Note that the figure will not
        be shown in this case (nilearn properties).
    fig_fname : str
        Path for saving figure if save_fig=True.
    """
    # center the x limits wrt the smaller extreme (minimum or maximum)
    v_abs = min(abs(cov['data'].min()), abs(cov['data'].max()))

    # plotting
    plt.figure()
    plt.imshow(cov.data, vmin=-v_abs, vmax=v_abs, cmap='RdBu')
    plt.title(title)
    if colorbar:
        plt.colorbar()

    # show figure if applicable
    if show_fig is True:
        plt.show()

    # saving
    if save_fig:
        if fig_fname is None:
            raise ValueError("Please give a figure name to save to.")
        plt.savefig(fig_fname, bbox_inches='tight')
