"""Functions related to decoding in source space

Jan. 2018

AUTHOR: Britta U. Westner <britta.wstnr[at]gmail.com
LICENCE: BSD 3-clause
"""
import time

from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel,
                          get_coef)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA


def sliding_logreg_source(X, y, cross_val, return_clf=False):
    """Run a sliding estimator with Logistic Regression on source data.

    Parameters:
    -----------
    X : np.array
        features.
    y : vector
        response vector.
    cross_val : cross validation object
        cross validation to adopt.
    return_clf : bool
        whether the clf object should be returned as well.

    Returns
    -------
    score : float
        cross-validated AUC score
    clf : classifier object
        If return_clf == True, the classifier object will be returned, too.
    """
    startt = time.time()

    # Model
    clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression()))
    sliding = SlidingEstimator(clf, scoring='roc_auc', n_jobs=1)

    print('Computing Logistic Regression.')
    score = cross_val_multiscore(sliding, X, y, cv=cross_val)

    endt = time.time()
    print('Done. Time elapsed for sliding estimator: %i seconds.'
          % (endt - startt))

    if return_clf is True:
        return score, clf
    else:
        return score


def get_pattern(X, y, clf, time_point=None):
    """Get pattern from classifier on X and y at peak time.

    Re-fit the classifier without cross-validation and get the patterns/
    coefficients.

    Parameters:
    -----------
    X : array
        The data to fit (features).
    y : vector
        The response vector.
    clf : sklearn classifier object
        The classifier to re-fit.
    time-point : float | None
        If data is more than two dimensional: which time-point to fit.

    Returns:
    --------
    pattern : array
        The sensor or source pattern of coefficients.
    """
    if time_point is not None:
        X = X[:, :, time_point]

    clf.fit(X, y)

    if clf.steps[-1][0] == 'logisticregression':
        pattern = get_coef(clf, 'coef_', inverse_transform=True)
    else:
        pattern = get_coef(clf, 'patterns_', inverse_transform=True)

    return pattern


def cca_covariance_matrices(x_cov, y_cov, n_comp=None):
    """Use CCA to align covariance matrices from different runs.

    Get a transform matrix to align covariance matrices with each other.

    Parameters:
    ----------
    x_cov : array
        covariance matrix that should be aligned
    y_cov : array
        covariance matrix that is used for alignment
    n_comp : int | None
        number of components
    """
    if n_comp is not None:
        tfm_model = CCA(n_components=n_comp)
    else:
        tfm_model = CCA(n_components=len(x_cov))
        # TODO: maybe rank is better

    x_al, y_al = tfm_model.fit(x_cov, y_cov).transform(x_cov)

    return x_al, y_al
