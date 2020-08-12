""" Functions to simulate brain activity.

AUTHOR: Britta U. Westner <britta.wstnr[at]gmail.com>
Credit: parts of the functions are inspired by:
https://github.com/kingjr/jr-tools
"""
import numpy as np

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
        envelope = np.exp(50. * -(times - 0.5 - trial) ** 2.)
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
        # Brownian noise:
        white_noise = np.random.randn(*evoked.data.shape)

        if noise_type == "white":
            evoked.data += (white_noise / snr)
        elif noise_type == "brownian":
            brownian_noise = np.cumsum(white_noise / snr, axis=1)
            evoked.data += brownian_noise
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
