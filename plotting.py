"""Assorted plotting functions.

Jan. 2018
"""
import numpy as np
import matplotlib.pyplot as plt
from mne import save_stc_as_volume
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img


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


def plot_source_act(stc, fwd, mri=None, threshold=None, thresh_ref=None,
                    title=None, timepoint=None, save_to_disk=False,
                    fig_fname=None):
    """Plot source activity on volume.

    Plots source activity on subject's MRI.
    NOTE: the function needs to save a .nii.gz file during for plotting. This
    file is saved into the current directiory as "tmp.nii.gz".

    Parameters:
    -----------
    stc : dict
        MNE Python beamformer output
    fwd : forward operator
        MNE forward model
    mri : string | None
        Can be path to a specific subject's brain or "mni" to plot on the MNI
        brain or False for not having any background image.
    threshold : float | 'auto' | None
        Threshold for plotting, if 'auto', nilearn's automatic threshold is
        used, if None, no thresholding is done.
    thresh_ref : string
        Reference for thresholding. Can be 'all' to use maximum across time and
        space or 'max_time' to use maximum time point or 'time' to refer to the
        time point given in timepoint.
    title : string | None
        Title for the figure.
    timepoint : float | string
        Time point that should be plotted. Can be given as float or can be
        'max' to select the time point with maximal activity.
    save_to_disk : bool
        whether the figure should be saved
    fig_fname : string
        where to save the figure to

    Returns
    -------
    nilearn figure.
    """
    nii_save_name = 'tmp.nii.gz'
    img = save_stc_as_volume(nii_save_name, stc, fwd['src'],
                             mri_resolution=False)

    if timepoint is 'max':
        vox, timepoint = np.unravel_index(stc.data.argmax(), stc.data.shape)

    if mri is None:
        mri = False  # this plots no brain at all
    elif mri is "mni":
        mri = None  # this hopefully plots the MNI brain

    if thresh_ref is 'all':
        threshold = np.max(stc.data) * threshold
    elif thresh_ref is 'max_time':
        if timepoint is not 'max':
            # in that case, maximum time point needs to be calculated now:
            _, m_tp = np.unravel_index(stc.data.argmax(), stc.data.shape)
        threshold = np.max(stc.data[:, m_tp]) * threshold

    fig = plot_stat_map(index_img(img, timepoint), bg_img=mri,
                        threshold=threshold, title=title)

    if save_to_disk:
        if fig_fname is None:
            raise ValueError("Please specify a file name to save figure to.")

        fig.savefig(fig_fname)


def plot_source_ts(stc, n_ts, abs=True, xlims=None, ylims=None, title=None):
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
    if ylims is not None:
        plt.ylim(ylims)
    plt.title(title)
    plt.show()


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
        Whether figure should be saved to disk.
    fig_fname : str
        Path for saving figure if save_fig=True.
    """

    # center the x limits wrt the smaller extreme (minimum or maximum)
    v_abs = min(abs(cov['data'].min()), abs(cov['data'].max()))

    # plotting
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
