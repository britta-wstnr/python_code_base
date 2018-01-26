"""Functions related to processing of sensor space data.

Author: bw
Jan. 2018"""
import mne
import numpy as np
from mne import find_events, Epochs
from mne.io import Raw
from bw_helper_functions import check_rank_cov_matrix


def read_run(raw_fname, run_num):
    """Read data from run run_num from disk.

    Reads both maxfiltered and raw data."""
    if '-sss.fif' in raw_fname:
        # maxfiltered data:
        run_fname = raw_fname.format(str(run_num))
        raw = Raw(run_fname, preload=True)
    else:
        # no maxfilter:
        run_fname = raw_fname.format(str(run_num))
        raw = Raw(run_fname, preload=True, allow_maxshield=True)

    return raw


def hilbert_epoching(raw, freqs, filt_bandwidth, baseline, t_win, mag=False,
                     resamp_rate):
    """Get data ready for hilbert beamformer.

    Filter, hilbertize, and epoch the data. Returns BP-filtered as well as
    hilbertized data.

    NOTE: This function has values specific to the experiment hard-coded,
    adjust before using for something else!!!

    Parameters:
    -----------
    raw : Raw
        raw data of one run.
    freqs : tuple or list (len=2)
        lower and upper limit for BP-filtering.
    filt_bandwidth : float
        transition bandwidth for FIR filter.
    baseline : tuple (len=2)
        baseline limits.
    t_win : tuple or list (len=2)
        time window for epoching.
    mag : bool
        whether only magnetometers should be considered, defaults to False.
    resamp_rate : float
        resampling rate, only hilbertized data will be resampled.

    Returns
    -------
    events :
        events used for epoching.
    epochs_filtered : Epochs
        bandpass-filtered epochs for covariance matrix estimation.
    epochs_hilbert : Epochs
        hilbertized epochs, resampled!.
    """
    # event business comes first - allows to get rid of stim channel later
    events = find_events(raw, stim_channel='STI101', shortest_event=1)
    sel = np.where(events[:, 2] <= 255)[0]
    events = events[sel, :]
    # Compensate for delay (as measured manually with photodiode)
    events[:, 0] += int(.050 * raw.info['sfreq'])

    # pick channels:
    if mag is True:
        raw.pick_types(meg='mag', eog=False, stim=False)
    else:
        raw.pick_types(meg=True, eog=False, stim=False)

    # filtering and hilbertizing
    print('Bandpass-filtering and hilbertizing for %i to %i Hz.' % freqs)

    raw.filter(freqs[0], freqs[1], n_jobs=1, l_trans_bandwidth=filt_bandwidth,
               h_trans_bandwidth=filt_bandwidth, fir_design='firwin')

    raw_hilbert = raw.copy().apply_hilbert(n_jobs=1, envelope=False)

    # epoching
    print('Epoching data.')
    epochs_filtered = Epochs(raw, events, proj=False, tmin=t_win[0],
                             tmax=t_win[1], baseline=baseline, preload=True)
    epochs_filtered._raw = None  # memory leakage

    epochs_hilbert = Epochs(raw_hilbert, events, proj=False, tmin=t_win[0],
                            tmax=t_win[1], baseline=baseline,
                            preload=True).resample(resamp_rate)
    epochs_hilbert._raw = None  # memory leakage

    return events, epochs_filtered, epochs_hilbert


def compute_covariance(epochs, t_win, noise=False, t_win_noise=None,
                       check=True, plot=False):
    """Compute data and noise covariance matrix.

    Computes the data covariance matrix and if desirable also the noise
    covariance matrix from the same data. Evaluates the goodness of the matrix
    and plots it.

    Parameters:
    -----------
    epochs : epochs
        epochs to compute the covariance matrices on.
    t_win : tuple or list
        time window to compute data covariance matrix on.
    noise : bool
        whether noise covariance matrix should be computed.
    t_win_noise : tuple or list
        time window to compute noise covariance matrix on.
    check : bool
        whether the covariance matrices' goodness should be checked.
    plot : bool
        whether the covariance matrices should be plotted.

    Returns
    -------
    data_cov : MNE CovMat
        data covariance matrix
    noise_cov : MNE CovMat
        noise covariance matrix, returns None if not computed.
    """
    data_cov = mne.compute_covariance(epochs, tmin=t_win[0], t_win[1])
    if check is True:
        check_rank_cov_matrix(data_cov, epochs)

    if noise is True:
        noise_cov = mne.compute_covaraince(epochs, tmin=t_win_noise[0],
                                           tmax=t_win_noise[1])
        if check is True:
            print('Noise covariance matrix:')
            check_rank_cov_matrix(noise_cov, epochs)

    # plotting business after info messages
    if plot is True:
        v_abs = max(abs((data_cov['data'].min(), data_cov['data'].max())))
        plt.imshow(data_cov.data, vmin=-v_abs, vmax=v_abs, cmap='RdBu')
        plt.title('Data covariance matrix.')
        plt.colorbar()
        plt.show()

        if noise is True:
            v_abs = max(abs((noise_cov['data'].min(),
                             noise_cov['data'].max())))
            plt.imshow(noise_cov.data, vmin=-v_abs, vmax=v_abs, cmap='RdBu')
            plt.title('Noise covariance matrix.')
            plt.colorbar()
            plt.show()

    # return whatever needs to be returned
    if noise is True:
        return data_cov, noise_cov
    else:
        return data_cov, None
