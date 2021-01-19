"""Functions for signal processing in the wider sense.

AUTHOR: Britta U. Westner <britta.wstnr[at]gmail.com>
"""
import mne
import numpy as np


def get_max_diff(evoked, evoked2=None, use_abs=False):
    """Get the maximum in difference curve between two evoked signals.

    Parameters:
    -----------
    evoked :  instance of evoked, array or list
        Can either be an instance of evoked (then evoked2 needs to be
        specified, too) or a list of evokeds.
    evoked2 : instance of evoked or array
        Needs to be specified if evoked is not a list
    use_abs : bool
        If the difference should be taken from the absolute signals,
        diserable e.g. for source signals.

    Returns
    -------
    evoked_diff : array
        The difference time curve.
    max_sens : int
        The sensor (or grid point) of the maximal difference between the
        curves.
    max_tp: int
        The time point of maximal difference.
    """
    # take care of evoked types and make them numpy array:
    if evoked2 is None:
        if type(evoked) is not list:
            raise ValueError("If evoked is not a list (got type %s), evoked2 "
                             "needs to be specified." % type(evoked))
        evoked1 = _check_type_evoked(evoked[0])
        evoked2 = _check_type_evoked(evoked[1])
    else:
        evoked1 = _check_type_evoked(evoked)
        evoked2 = _check_type_evoked(evoked2)

    # get the difference
    if use_abs:
        diff = np.abs(evoked1) - np.abs(evoked2)
    else:
        diff = evoked1 - evoked2

    max_sens, max_tp = np.unravel_index(np.abs(diff).argmax(), diff.shape)

    return diff, max_sens, max_tp


def _check_type_evoked(evoked):
    """Make sure type of evoked is either MNE type or numpy array."""
    if type(evoked) is mne.evoked.EvokedArray:
        evoked = evoked.data
    elif type(evoked) is not np.ndarray:
        raise ValueError("Do not know type of evoked.")
    return evoked


def estimate_snr(epochs, active, baseline):
    """Compute effective SNR of the data."""
    rms_signal = np.sqrt(np.mean(epochs._data[:, :,
                                              active[0]:active[1]] ** 2, 2))
    rms_noise = np.sqrt(np.mean(epochs._data[:, :,
                                             baseline[0]:baseline[1]] ** 2, 2))

    est_snrs = np.mean(rms_signal, 0) / np.mean(rms_noise, 0)

    return 20 * np.log10(est_snrs)
