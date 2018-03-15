""" Preliminary way of storing various functions re Decoding project.

Should eventually be organized better.

Jan. 2018
"""
from mne.utils import estimate_rank
from numpy.linalg import cond
from decimal import Decimal


def check_rank_cov_matrix(cov_mat, epochs):
    """Check the covariance matrix and report rank and channel number."""
    ch_num = len(cov_mat.ch_names)
    rank_cov = estimate_rank(cov_mat['data'], tol='auto')
    cov_ok = ch_num - rank_cov
    cond_num = cond(cov_mat['data'])
    if cov_ok > 0 and cond_num > 10e15:
        # cond num: amazing is < 10e4, really shitty is > 10e12, horrible 10e19
        print_ok = 'WARNING! Bad cov matrix.'
    else:
        print_ok = 'ok.'

    print('%i trials, %i channels, cov matrix has rank %i and condition '
          'number %s: %s'
          % (len(epochs.events), ch_num, rank_cov,
             '{:.2E}'.format(Decimal(cond_num)), print_ok))


def check_versions(package_list):
    """Print the versions of packages."""
    for package in package_list:
        print(package.__name__, package.__version__)
