"""Functions related to decoding in sensor space

Author: bw
Jan. 2018"""
import time

from mne.decoding import SlidingEstimator, cross_val_multiscore, LinearModel

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sliding_logreg_source(X, y, cross_val):
    """Run a sliding estimator with Logistic Regression on source data.
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

    return score
