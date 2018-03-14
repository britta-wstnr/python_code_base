"""Functions to prepare and run beamformers.

Author: bw
Jan. 2018 - Mar. 2018"""
import mne
import numpy as np
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_epochs


def compute_grid(subject, bem_name, t1_fname=None, volume_grid=True,
                 spacing=10., read_from_disk=False, src_fname=None,
                 save_to_disk=False):
    """Compute source grid (src).

    Reads or computes source space or surface grid.

    Parameters:
    -----------
    subject : string
        subject name.
    bem_name : path
        path to BEM model.
    t1_fname : path
        path to MRI, needed for source space model.
    volume_grid : bool
        whether volume source model should be computed.
    spacing : float
        source spacing for volume grid or surface grid, defaults to 10 mmm for
        volume source space and "oct6" for surface space. Needs to be float
        for volume source spaces and string for surface spaces.
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
                                                subjects_dir=subject)
        else:  # surface grid
            if spacing is None:
                spacing = "oct6"
            elif isinstance(spacing, str) is not True:
                raise ValueError("spacing needs to be given as a string for "
                                 "surface source models.")

            src = mne.setup_source_space(subject, spacing=spacing,
                                         subjects_dir=subject, add_dist=False)
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
                    pick_ori='max-power', weight_norm='nai', return_stc=False):
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
    return_stc : bool
        whether the stcs list should be returned in addition to the matrix.

    Returns
    -------
    stcs_mat : numpy array
        matrix with all source trials
    filters : dict
        spatial filter used in computation
    stcs : list of MNE stcs
        original output of apply_lcmv_epochs, None if return_stc=False
    """
    filters = make_lcmv(epochs.info, fwd, data_cov=data_cov,
                        noise_cov=noise_cov, pick_ori=pick_ori, reg=reg,
                        weight_norm=weight_norm)

    # apply that filter to epochs
    stcs = apply_lcmv_epochs(epochs, filters, return_generator=False,
                             max_ori_out='signed')

    # get matrix
    for_stack = []
    for stc in stcs:
        for_stack.append(np.abs(stc.data))

    stcs_mat = np.stack(for_stack, axis=0)

    if return_stc is True:
        return stcs_mat, filters, stcs
    else:
        return stcs_mat, filters, None
