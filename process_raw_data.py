"""Functions related to processing of sensor space data.

Jan. - Mar. 2018

AUTHOR: Britta U. Westner
LICENCE: BSD 3-clause
"""
import mne
import numpy as np

from mne import find_events, Epochs
from mne.io import Raw
from helper_functions import check_rank_cov_matrix


def read_run(raw_fname, run_num):
    """Read data from run run_num from disk.

    Reads both maxfiltered and raw data.

    Parameters:
    -----------
    raw_fname : string
        path to the raw file with place holder for the run number
    run_num : int
        run number to be filled into the path

    Returns
    -------
    raw : Raw object
    """
    if '-sss.fif' in raw_fname:
        # maxfiltered data:
        run_fname = raw_fname.format(str(run_num))
        raw = Raw(run_fname, preload=True)
    else:
        # no maxfilter:
        run_fname = raw_fname.format(str(run_num))
        raw = Raw(run_fname, preload=True, allow_maxshield=True)

    return raw


def filter_epoching(raw, freqs, filt_transwidth, baseline, t_win,
                    resamp_rate=None, mag=False):
    """Filter and epoch data for ERP analysis.

    Filter, epoch and eventually downsample data.

    NOTE: This function has values specific to the experiment hard-coded,
    adjust before using for something else!!!

    Parameters:
    -----------
    raw : Raw
        raw data of one run.
    freqs : tuple or list (len=2)
        lower and upper limit for filtering (successive HP and LP filter)
    filt_transwidth : tuple or list (len=2)
        transition bandwidth for FIR filters (HP and LP).
    baseline : tuple (len=2)
        baseline limits.
    t_win : tuple or list (len=2)
        time window for epoching.
    resamp_rate : None | float
        resampling rate, if not None, data will be resampled
    mag : bool
        whether only magnetometers should be considered, defaults to False.

    Returns
    -------
    events :
        events used for epoching.
    epochs_filtered : Epochs
        filtered epochs
    """
    # event business comes first - allows to get rid of stim channel later
    # TODO: make this smarter and easier to control
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

    # filtering
    (str(kk) for kk in freqs)  # handle the possibility of None type
    print('Filtering from %s to %s Hz.' % (freqs[0], freqs[1]))

    raw.filter(freqs[0], freqs[1], n_jobs=1,
               l_trans_bandwidth=filt_transwidth[0],
               h_trans_bandwidth=filt_transwidth[1],
               fir_design='firwin')

    # epoching
    print('Epoching data.')
    epochs_filtered = Epochs(raw, events, proj=False, tmin=t_win[0],
                             tmax=t_win[1], baseline=baseline, preload=True)
    epochs_filtered._raw = None  # memory leakage

    if resamp_rate is not None:
        print('Resampling data at %i Hz.' % resamp_rate)
        epochs_filtered.resample(resamp_rate)

    return events, epochs_filtered


def hilbert_epoching(raw, freqs, filt_transwidth, baseline, t_win, resamp_rate,
                     mag=False):
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
    filt_transwidth : float
        transition bandwidth for FIR filter.
    baseline : tuple (len=2)
        baseline limits.
    t_win : tuple or list (len=2)
        time window for epoching.
    resamp_rate : float
        resampling rate, only hilbertized data will be resampled.
    mag : bool
        whether only magnetometers should be considered, defaults to False.

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
    # TODO: make this smarter and easier to control
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
    print('Bandpass-filtering and hilbertizing for %i to %i Hz.'
          % (freqs[0], freqs[1]))

    raw.filter(freqs[0], freqs[1], n_jobs=1, l_trans_bandwidth=filt_transwidth,
               h_trans_bandwidth=filt_transwidth, fir_design='firwin')

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

    Returns
    -------
    data_cov : MNE CovMat
        data covariance matrix
    noise_cov : MNE CovMat
        noise covariance matrix, returns None if not computed.
    """
    data_cov = mne.compute_covariance(epochs, tmin=t_win[0], tmax=t_win[1])
    if check is True:
        check_rank_cov_matrix(data_cov, epochs)

    if noise is True:
        noise_cov = mne.compute_covariance(epochs, tmin=t_win_noise[0],
                                           tmax=t_win_noise[1])
        if check is True:
            print('Noise covariance matrix:')
            check_rank_cov_matrix(noise_cov, epochs)

    # return whatever needs to be returned
    if noise is True:
        return data_cov, noise_cov
    else:
        return data_cov, None


def compute_snr(evoked, t_baseline, t_signal, label):
    """Computes the signal-to-noise ratio on an evoked signal.

    Computes the signal-to-noise ratio (SNR) on an evoked signal, using the
    root-mean-square.

    Parameters:
    -----------
    evoked : evoked
        evoked data to compute SNR on
    t_baseline : tuple or list
        time window to use as noise.
    t_signal : tuple or list
        time window to use as signal.
    label : list of integers
        the channels to include in the computation.

    Returns
    -------
    snr : float
        signal-to-noise ratio
    """
    bl = evoked.time_as_index(t_baseline)
    sig = evoked.time_as_index(t_signal)

    rms_bl = np.sqrt(np.mean(evoked.data[label, bl[0]:bl[1]] ** 2))
    rms_sig = np.sqrt(np.mean(evoked.data[label, sig[0]:sig[1]] ** 2))

    snr = ((rms_sig - rms_bl) / rms_bl)
    return snr
