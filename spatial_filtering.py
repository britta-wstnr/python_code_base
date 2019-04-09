"""Functions to prepare and run beamformers.

AUTHOR: Britta U. Westner <britta.wstnr[at]gmail.com>
"""
import mne
import numpy as np
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_epochs
from matrix_transforms import get_coord_from_peak, get_distance


def compute_grid(subject, subjects_dir, bem_name, t1_fname=None,
                 volume_grid=True, spacing=None, read_from_disk=False,
                 src_fname=None, save_to_disk=False):
    """Compute source grid (src).

    Reads or computes source space or surface grid.

    Parameters:
    -----------
    subject : string
        subject name.
    subjects_dir : string | None
        subject directory, can be None if same as subject.
    bem_name : path
        path to BEM model.
    t1_fname : path
        path to MRI, needed for source space model.
    volume_grid : bool
        whether volume source model should be computed.
    spacing : float | string | None
        source spacing for volume grid or surface grid, defaults to 10 mmm for
        volume source space and "oct6" for surface space if set to None. Needs
        to be float for volume source spaces and string for surface spaces.
    read_from_disk : bool
        if True, read pre-computed grid from disk.
    src_fname : path
        path to grid on disk, either for saving or reading.
    save_to_disk : bool
        whether grid should be saved to disk.

    Returns
    -------
    src : dict
        MNE source model.
    """
    if subjects_dir is None:
        subjects_dir = subject

    if read_from_disk is True:
        src = mne.read_source_spaces(src_fname)
    else:
        if volume_grid:
            if spacing is None:
                spacing = 10.
            elif isinstance(spacing, float) is not True:
                raise ValueError("spacing needs to be given as a float for "
                                 "volume source models.")
            src = mne.setup_volume_source_space(pos=spacing, mri=t1_fname,
                                                bem=bem_name,
                                                subjects_dir=subjects_dir)
        else:  # surface grid
            if spacing is None:
                spacing = "oct6"
            elif isinstance(spacing, str) is not True:
                raise ValueError("spacing needs to be given as a string for "
                                 "surface source models.")

            src = mne.setup_source_space(subject, spacing=spacing,
                                         subjects_dir=subjects_dir,
                                         add_dist=False)
        if save_to_disk:
            mne.write_source_spaces(src_fname, src, overwrite=True)

    return src


def compute_forward(info, bem, src, trans_fname, read_from_disk=False,
                    fwd_fname=None, save_to_disk=False):
    """Compute forward solution.

    Reads or computes forward solution.

    Parameters:
    -----------
    info : dict
        info from epoch, contains channel positions etc.
    bem : headmodel or path
        BEM model or path to BEM model.
    src : source space
        source grid.
    trans_fname : path
        path to transformation matrix.
    read_from_disk : bool
        if True, read pre-computed fwd model from disk.
    fwd_fname : path
        path to fwd model on disk, either for saving or reading.
    save_to_disk : bool
        whether fwd model should be saved to disk.

    Returns
    -------
    fwd : dict.
        MNE forward operator
    """
    if read_from_disk is True:
        fwd = mne.read_forward_solution(fwd_fname)
    else:
            fwd = mne.make_forward_solution(info, trans=trans_fname,
                                            src=src, bem=bem, meg=True,
                                            eeg=False, n_jobs=1)

            if save_to_disk is True:
                mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

    return fwd


def run_lcmv_evoked(evoked, fwd, data_cov, reg, noise_cov=None,
                    pick_ori='max-power', weight_norm='nai'):
    """Run LCMV on average.

    Run weight-normalized LCMV beamformer on evoked data, will return an stc
    object.

    Parameters:
    -----------
    evoked : MNE evoked
        evoked data to source reconstruct.
    fwd : MNE forward model
        forward model.
    data_cov : MNE covariance estimate
        data covariance matrix.
    reg : float
        regularization parameter
    noise_cov : MNE covariance estimate
        noise covariance matrix, optional

    Returns
    -------
    stc : MNE stcs
        original output of apply_lcmv
    filters : dict
        spatial filter used in computation
    """
    filters = make_lcmv(evoked.info, fwd, data_cov=data_cov,
                        noise_cov=noise_cov, pick_ori=pick_ori, reg=reg,
                        weight_norm=weight_norm)

    # apply that filter to epochs
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')

    return stc, filters


def run_lcmv_epochs(epochs, fwd, data_cov, reg, noise_cov=None,
                    pick_ori='max-power', weight_norm='nai', verbose=False):
    """Run LCMV on epochs.

    Run weight-normalized LCMV beamformer on epoch data, will return matrix of
    trials or stc object.

    Parameters:
    -----------
    epochs : MNE epochs
        epochs to source reconstruct.
    fwd : MNE forward model
        forward model.
    data_cov : MNE covariance estimate
        data covariance matrix
    reg : float
        regularization parameter
    noise_cov : MNE covariance estimate
        noise covariance matrix, optional
    verbose : bool
        overrides default verbose level, defaults to False, i.e., no logger
        info.

    Returns
    -------
    stcs_mat : numpy array
        matrix with all source trials
    stc : MNE stc
        single trial stc object (last trial)
    filters : dict
        spatial filter used in computation
    """
    filters = make_lcmv(epochs.info, fwd, data_cov=data_cov,
                        noise_cov=noise_cov, pick_ori=pick_ori, reg=reg,
                        weight_norm=weight_norm, verbose=verbose)

    # apply that filter to epochs
    stcs = apply_lcmv_epochs(epochs, filters, return_generator=True,
                             max_ori_out='signed', verbose=verbose)

    # preallocate matrix
    stcs_mat = np.ones((epochs._data.shape[0], fwd['nsource'],
                        len(epochs.times)))

    if verbose is False:
        mne.set_log_level('WARNING')

    # resolve generator
    for trial in range(epochs._data.shape[0]):
        # last time: also save stc
        if trial == 0:
            stc = next(stcs)
            stcs_mat[trial, :, :] = stc.data
        else:
            stcs_mat[trial, :, :] = next(stcs).data

    return stcs_mat, stc, filters


def compute_activity_spread(stc, fwd, threshold=0.8):
    """Compute a weighted spread index for source activity.

    Compute an index for the spread of source activity, weighted by the
    distance of voxels to the maximum of the source. Only considers activity
    at the timepoint of the maximum.

    Parameters:
    -----------
    stc : MNE source object
        source to compute index on.
    fwd : MNE forward model
        forward model.
    threshold : float
        the threshold for voxels to consider (% of maximum activity).

    Returns
    -------
    index : float
        the computed weighted spread index.
    """
    max_val = stc.data.max()
    vox, time_point = np.unravel_index(stc.data.argmax(), stc.data.shape)
    max_coord = get_coord_from_peak(stc, fwd)
    cut_val = max_val * threshold
    voxels = np.where(stc.data[:, time_point] >= cut_val)

    spread = 0.
    for vox_ii in voxels[0]:
        if vox_ii == vox:
            spread = spread + 1.
        else:
            coord = fwd['src'][0]['rr'][stc.vertices[vox_ii], ]
            dist = get_distance(max_coord, coord)
            weighted_dist = 1. / (dist * 1000)  # convert in mm
            # weighted_act = stc.data[vox_ii, time_point] / max_val
            spread = spread + weighted_dist * weighted_dist

    return spread


def source2epoch(stcs_mat, grid_num, sfreq):
    """Transform source data matrix to EpochArray ("virtual channels").

    Use source space (or other) data that are present as a matrix and build a
    Epoch object from that to enable all epoch-based processing.

    Parameters:
    -----------
    stcs_mat : matrix
        data matrix with dimensions [trials, grid points, time points]
    grid_num : int
        number of grid points
    sfreq : float
        sampling frequency

    Returns:
    --------
    epochs : mne.EpochsArray object
        the data wrapped in an epochs object
    """
    ch_names = [str(x) for x in range(grid_num)]
    ch_types = ['mag'] * grid_num

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    epochs = mne.EpochsArray(stcs_mat, info, tmin=0., verbose=False)

    return epochs
