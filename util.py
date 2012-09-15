from __future__ import division

from pyhsmm.util.general import sample_discrete_from_log

def sample_log_weighted_list(lst):
    ks,weights = zip(*lst)
    return ks[sample_discrete_from_log(weights)]

