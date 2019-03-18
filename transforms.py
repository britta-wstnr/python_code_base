"""Transform functions for use in scikit-learn pipelines

AUTHOR: Britta U. Westner <britta.wstnr[at]gmail.com>
"""
import mne
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class lcmvEpochs(TransformerMixin, BaseEstimator):
    def __init__(self, info, fwd, t_win, t_win_noise, reg,
                 pick_ori='max-power',
                 weight_norm='nai'):
        self.info = info
        self.fwd = fwd
        self.t_win = t_win
        self.t_win_noise = t_win_noise
        self.reg = reg
        self.pick_ori = pick_ori
        self.weight_norm = weight_norm

    def fit(self, X, y):
        from mne.beamformer import make_lcmv
        from process_raw_data import compute_covariance

        epochs = mne.EpochsArray(X, self.info, verbose=False)
        self.data_cov, self.noise_cov = compute_covariance(
            epochs, t_win=self.t_win, noise=True,
            t_win_noise=self.t_win_noise, check=False, plot=False)
        self.filters = make_lcmv(self.info, self.fwd, self.data_cov,
                                 noise_cov=self.noise_cov,
                                 pick_ori=self.pick_ori,
                                 weight_norm=self.weight_norm)

        return self

    def transform(self, X, **transform_params):
        from mne.beamformer import apply_lcmv_epochs
        mne.set_log_level('WARNING')
        epochs = mne.EpochsArray(X, self.info, verbose=False)
        stcs = apply_lcmv_epochs(epochs, self.filters,
                                 return_generator=True,
                                 max_ori_out='signed', verbose=False)
        stcs_mat = np.ones((X.shape[0], self.fwd['nsource'],
                           X.shape[2]))
        for trial in range(X.shape[0]):
            stcs_mat[trial, :, :] = next(stcs).data

        # stcs_mat is [trials, grid points, time points]
        return np.mean((stcs_mat ** 2), axis=2)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
