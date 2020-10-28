""" Functions to simulate brain activity.

AUTHOR: Britta U. Westner <britta.wstnr[at]gmail.com>
Credit: parts of the functions are inspired by:
https://github.com/kingjr/jr-tools
"""
import numpy as np
import pandas as pd

from mne.forward import apply_forward
from mne.simulation import simulate_sparse_stc


def generate_signal(times, freq, n_trial=1, phase_lock=False):
    """Simulate a time series.

    Parameters:
    -----------
    times : np.array
        time vector
    freq : float
        frequency of oscillations
    n_trial : int
        number of trials, defaults to 1.
    """
    signal = np.zeros_like(times)

    for trial in range(n_trial):
        envelope = np.exp(1000. * -(times - 0.5 - trial) ** 2.)
        if phase_lock is False:
            phase = np.random.rand() * 2 * np.pi
            signal += np.cos(phase + freq * 2 * np.pi * times) * envelope
        else:
            signal += np.cos(freq * 2 * np.pi * times) * envelope
    return signal * 1e-12


def simulate_evoked_osc(info, fwd, n_trials, freq, label, loc_in_label=None,
                        picks=None, loc_seed=None, snr=None,
                        noise_type="white", return_matrix=True,
                        filtering=None, phase_lock=False):
    """Simulate evoked oscillatory data based on a given fwd model and dipole.

    Parameters:
    -----------
    info : MNE info object
        data info, e.g., from raw
    fwd : MNE forward object
        forward model object
    freq : float
        freq of simulated oscillation
    n_trials : int
        number of trials
    label : MNE label
        source space label to simulate data in
    loc_in_label : None | int
        Specify the random generator state for dipole simulation within the
        label. Defaults to np.random.RandomState if None.
    picks : None | string
        Channel types to pick from evoked, can be 'mag' or 'grad'. None
        defaults to all.
    seed : None | int
        Seed for the time series simulation, only relevant for location in
        label.
    snr : None | float
        If not None, signal-to-noise ratio for resulting signal (adding noise).
    noise_type : str
        Type of noise. Supported is at the moment: "white" and "brownian".
    return_matrix : bool
        If True, a matrix of epochs will be returned and the evoked object will
        be averaged across trials.
    filtering : None | dict
        If None (default), no filtering is done. If filtering should be done,
        the dictionary needs to contain the following keys:
            "hp" : high pass cutoff, float.
            "lp" : low pass cutoff, float.
            "fir_design" : FIR design, string, see evoked.filter()
            "lp_tw" : transition width for low pass, float, optional.
            "hp_tw" : transition width for high pass, float, optional.
    phase_lock : bool
        If True, the oscillation will be phase-locked across trials.

    Returns:
    --------
    evoked : MNE evoked object
        Simulated sensor data.
    stc : MNE source time course object
        Simulated source space data.
    epochs : np.array
        Matrix with epochs, if return_matrix is True.
    """
    if loc_seed is not None:
        np.random.seed(loc_seed)
    if loc_in_label is None:
        loc_in_label = np.random.RandomState()

    np.random.seed()  # reset to random seed to not get funky results for noise

    times = np.arange(0., n_trials, 1./info['sfreq'])
    stc = simulate_sparse_stc(fwd['src'], n_dipoles=1, times=times,
                              random_state=loc_in_label, labels=label,
                              data_fun=lambda
                              times: generate_signal(times, freq, n_trials,
                                                     phase_lock=phase_lock))

    # go to sensor space
    evoked = apply_forward(fwd, stc, info, verbose=False, use_cps=False)

    # pick channel types if applicable
    if picks is not None:
        evoked.pick_types(meg=picks)

    # take care of SNR:
    evoked.data /= np.std(evoked.data)

    if snr is not None:
        if snr == 0.0:
            raise ValueError('You asked me to divide by 0. Please change '
                             'snr parameter.')

        if noise_type == "white":
            white_noise = np.random.randn(*evoked.data.shape)
            evoked.data += (white_noise / snr)
        elif noise_type == "brownian":
            white_noise = np.random.randn(*evoked.data.shape)
            brownian_noise = np.cumsum(white_noise / snr, axis=1)
            evoked.data += brownian_noise
        elif noise_type == "pink":
            pink_noise = make_pink_noise(evoked.data.shape[1], 10,
                                         evoked.data.shape[0])
            evoked.data += (pink_noise / snr)
        else:
            raise ValueError('So far, only white and brownian noise is '
                             'implemented, got %s' % noise_type)

        # evoked.data *= 1e-12

    if filtering is not None:
        if "lp_tw" not in filtering:
            filtering["lp_tw"] = "auto"
        if "hp_tw" not in filtering:
            filtering["hp_tw"] = "auto"

        evoked.filter(filtering["hp"], filtering["lp"],
                      fir_design=filtering["fir_design"],
                      l_trans_bandwidth=filtering["hp_tw"],
                      h_trans_bandwidth=filtering["lp_tw"],
                      verbose=False)

    # take care of trials:
    if return_matrix is True:
        epochs = evoked._data
        epochs = epochs.reshape([len(evoked.ch_names),
                                 n_trials, -1]).transpose(1, 0, 2)

        evoked.crop(0., evoked.times[int((times.shape[0] / n_trials) - 1)])
        evoked._data[:, :] = epochs.mean(axis=0)

        return evoked, stc, epochs

    else:
        return evoked, stc


def make_pink_noise(noise_samples, n_changes, n_sensors):
    """Generate pink noise following the Voss algorithm.

    This function simulates pink (1/f) noise following the algorithm by
    Voss & Clarke (1978), J Acoust Soc.
    The implementation follows a blog post by Allen Downey
    https://www.dsprelated.com/showarticle/908.php

    Parameters:
    -----------
    noise_samples : int
        number of samples to put out (length of signal)
    n_changes : int
        How often the random source changes. The bigger the number, the more
        low frequency content is in the signal.
    n_sensors : int
        How many different noise signals should be generated (i.e. how many
        sensors). This will correspond to the rows of the output.

    Returns:
    --------
    noise_signal : array
        Pink noise of the dimensions sensors x samples.
    """

    noise_signal = np.empty((n_sensors, noise_samples))

    for ii, sig in enumerate(noise_signal):
        array = np.empty((noise_samples, n_changes))
        array.fill(np.nan)
        array[0, :] = np.random.random(n_changes)
        array[:, 0] = np.random.random(noise_samples)

        cols = np.random.geometric(0.5, noise_samples)
        cols[cols >= n_changes] = 0
        rows = np.random.randint(noise_samples, size=noise_samples)
        array[rows, cols] = np.random.random(noise_samples)

        df = pd.DataFrame(array)
        df.fillna(method='ffill', axis=0, inplace=True)
        noise_signal[ii, :] = df.sum(axis=1)

    return noise_signal
