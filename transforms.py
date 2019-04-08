"""Transform functions for use in scikit-learn pipelines

AUTHOR: Britta U. Westner <britta.wstnr[at]gmail.com>
"""
import mne
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class lcmvEpochs(TransformerMixin, BaseEstimator):
    def __init__(self, info, fwd, t_win, t_win_noise, tmin, reg,
                 pick_ori='max-power',
                 weight_norm='nai',
                 erp=False, time_idx=None):
        self.info = info
        self.fwd = fwd
        self.t_win = t_win
        self.t_win_noise = t_win_noise
        self.reg = reg
        self.tmin = tmin
        self.pick_ori = pick_ori
        self.weight_norm = weight_norm
        self.erp = erp
        self.time_idx = time_idx

    def fit(self, X, y):
        from mne.beamformer import make_lcmv
        from process_raw_data import compute_covariance

        epochs = mne.EpochsArray(X, self.info, tmin=self.tmin, verbose=False)
        self.data_cov, self.noise_cov = compute_covariance(
            epochs, t_win=self.t_win, noise=True,
            t_win_noise=self.t_win_noise, check=False, plot=False)
        self.filters = make_lcmv(self.info, self.fwd, self.data_cov,
                                 noise_cov=self.noise_cov,
                                 pick_ori=self.pick_ori,
                                 weight_norm=self.weight_norm)

        return self

    def transform(self, X):
        from mne.beamformer import apply_lcmv_epochs
        mne.set_log_level('WARNING')
        epochs = mne.EpochsArray(X, self.info, tmin=self.tmin, verbose=False)
        stcs = apply_lcmv_epochs(epochs, self.filters,
                                 return_generator=True,
                                 max_ori_out='signed', verbose=False)
        stcs_mat = np.ones((X.shape[0], self.fwd['nsource'],
                           X.shape[2]))
        for trial in range(X.shape[0]):
            stcs_mat[trial, :, :] = next(stcs).data

        # stcs_mat is [trials, grid points, time points]
        if self.erp is False:
            return np.mean((stcs_mat ** 2), axis=2)
        else:
            return np.squeeze(stcs_mat[:, :, self.time_idx])

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class lcmvHilbert(TransformerMixin, BaseEstimator):
    def __init__(self, info, fwd, t_win, t_win_noise, tmin, reg,
                 pick_ori='max-power', weight_norm='nai', power_win=None):
        self.info = info
        self.fwd = fwd
        self.t_win = t_win
        self.t_win_noise = t_win_noise
        self.tmin = tmin
        self.reg = reg
        self.pick_ori = pick_ori
        self.weight_norm = weight_norm
        self.power_win = power_win

    def fit(self, X, y):
        from mne.beamformer import make_lcmv
        from process_raw_data import compute_covariance

        epochs = mne.EpochsArray(X, self.info, tmin=self.tmin, verbose=False)
        self.data_cov, self.noise_cov = compute_covariance(
            epochs, t_win=self.t_win, noise=True,
            t_win_noise=self.t_win_noise, check=True, plot=False)
        self.filters = make_lcmv(self.info, self.fwd, self.data_cov,
                                 noise_cov=self.noise_cov,
                                 pick_ori=self.pick_ori,
                                 weight_norm=self.weight_norm)
        return self

    def transform(self, X):
        from scipy import signal
        from mne.beamformer import apply_lcmv_epochs
        mne.set_log_level('WARNING')
        hilbert_X = np.abs(signal.hilbert(X))
        epochs = mne.EpochsArray(hilbert_X, self.info, verbose=False)
        stcs = apply_lcmv_epochs(epochs, self.filters, return_generator=True,
                                 max_ori_out='signed', verbose=False)
        stcs_mat = np.ones((X.shape[0], self.fwd['nsource'],
                           X.shape[2]))
        for trial in range(X.shape[0]):
            stcs_mat[trial, :, :] = next(stcs).data

        # stcs_mat is [trials, grid points, time points]
        if self.power_win is None:
            self.power_win = self.t_win
        time_idx = epochs.time_as_index(self.power_win)

        return np.sum(stcs_mat[:, :, time_idx[0]:time_idx[1]] ** 2, axis=2)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
