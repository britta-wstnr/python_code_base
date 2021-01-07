""" Functions to generate data.

AUTHOR: Britta U. Westner <britta.wstnr[at]gmail.com>
Credit: parts of the functions are inspired by:
 https://github.com/kingjr/jr-tools
"""
import numpy as np

# mne
import mne
from mne import read_forward_solution
from mne.cov import compute_covariance
from mne.datasets import sample

import nibabel as nib
import random

# my own scripts
from process_raw_data import compute_covariance  # noqa
from matrix_transforms import stc_2_mgzvol  # noqa
from plotting import plot_source_act  # noqa
from simulations import generate_signal, simulate_evoked_osc  # noqa
from source_space_decoding import get_pattern  # noqa
from spatial_filtering import run_lcmv_epochs  # noqa
from transforms import lcmvEpochs  # noqa


def generate_data(label_names, n_trials, freqs, snr, pred_filter=True,
                  phase_lock=False, filt_def=None, loc='center'):
    """Simulate the data for the source decoding project.

    Possibility to add noise on sensor level.
    Simulations are based on the SAMPLE data set.
    Be aware of several hard coded options since geared towards specific
    project."""

    # load specific stuff from the SAMPLE data set
    data_path = sample.data_path() + '/MEG/sample/'
    fwd_sim = read_forward_solution(data_path +
                                    'sample_audvis-meg-eeg-oct-6-fwd.fif',
                                    verbose=False)
    # MRI
    t1_fname = data_path + '../../subjects/sample/mri/T1.mgz'
    mri_mgz = nib.load(t1_fname)
    info = mne.io.read_info(data_path + 'sample_audvis-no-filter-ave.fif',
                            verbose=False)
    info['sfreq'] = 250.

    # labels
    label_l = mne.read_label(data_path + 'labels/' + label_names[0])
    label_r = mne.read_label(data_path + 'labels/' + label_names[1])
    labels = [label_l, label_r]
    if loc == 'center':
        label_l.values[:] = 1.
        label_r.values[:] = 1.
        center_l = label_l.center_of_mass(subject='sample')
        center_r = label_r.center_of_mass(subject='sample')
        locations = (center_l * 3, center_r * 3)  # 3 orientations
    elif loc == 'random':
        locations = (random.choice(range(len(label_l) * 3)),
                     random.choice(range(len(label_r) * 3)))
    else:
        raise ValueError('loc has to be "random" or "center", got %s.' % loc)

    # simulate the data
    X = []
    sim_coords = []
    evokeds = []
    mu = None
    for label, loc, freq in zip(labels, locations, freqs):

        if pred_filter is True:
            if filt_def is None:
                raise ValueError("Specify filtering boundaries.")
            else:
                filtering = {"hp": filt_def[0], "lp": filt_def[1],
                             "fir_design": "firwin"}
        else:
            filtering = None

        evoked, stc, x_loc, mu = simulate_evoked_osc(info, fwd_sim,
                                                     n_trials=n_trials,
                                                     freq=freq, label=[label],
                                                     loc_in_label=loc,
                                                     picks='mag',
                                                     snr=snr, mu=mu,
                                                     loc_seed=loc,
                                                     return_matrix=True,
                                                     filtering=filtering,
                                                     phase_lock=phase_lock,
                                                     noise_type='white')
        # data
        X.append(x_loc)

        # coordinates
        vert_tmp = [stc.vertices[x] for x in [0, 1]
                    if stc.vertices[x].size != 0]
        src_coords_tmp = [fwd_sim['src'][x]['rr'] for x in [0, 1]
                          if stc.vertices[x].size != 0]
        coord_sim = src_coords_tmp[0][vert_tmp[0]]
        coord_mgz = stc_2_mgzvol(coord_sim[0], fwd_sim, mri_mgz)
        sim_coords.append(coord_mgz)
        evokeds.append(evoked)

    # get X and y
    X = np.vstack(X)
    y = np.r_[np.zeros(n_trials), np.ones(n_trials)]

    return X, y, sim_coords, evokeds


def get_power_or_erp(X, data_obj, phase_lock, power_win, time_point=0.):

    time_idx = data_obj[0].time_as_index(time_point)
    power_idx = data_obj[0].time_as_index(power_win)

    if phase_lock is True:
        # we choose one time_point
        time_idx = data_obj[0].time_as_index(time_point)
        X_out = X[:, :, time_idx[0]]
    else:
        # power
        X_out = np.mean(X[:, :, power_idx[0]:power_idx[1]]**2, axis=2)

    return X_out, time_idx
