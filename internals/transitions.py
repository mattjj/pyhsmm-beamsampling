from __future__ import division
import numpy as np

from pyhsmm.util.general import rle
from collections import defaultdict
import itertools, operator

class Beta(object):
    '''
    An implementation of the GEM-distributed beta vector for an HDP(-H(S)MM).

    Can call __getitem__ at any index to get beta's value at that index;
    values are lazily generated and 'size biased' according to the order in
    which indices are queried (so to be size biased over (1,2,3,...), those
    indices should be queried first). Indices themselves have no meaning and can
    be arbitrary hashables, so there's no need to use positve integers in
    counting order.

    Can call resample(countsdict), where countsdict maps from indices
    to observation counts.
    '''
    def __init__(self,gamma_0):
        self.gamma_0 = gamma_0
        self._remaining = 1.
        self._new_betavec()

    def resample(self,countsdict):
        weights = np.random.dirichlet(np.array(countsdict.values()+[self.gamma_0]))
        self._new_betavec()
        self._betavec.update(zip(countsdict.keys(),weights[:-1]))
        self._remaining = weights[-1]

    def __getitem__(self,k):
        return self._betavec[k]

    def _stickbreaking(self,gamma):
        while True:
            p = np.random.beta(1,gamma)
            piece, self._remaining = p*self._remaining, (1-p)*self._remaining
            yield piece

    def _new_betavec(self):
        self._betavec = defaultdict(self._stickbreaking(self.gamma_0).next)


class HDPHMMPiRow(object):
    '''
    An implementation of the rows of the transition matrix for BEAM-SAMPLING
    inference in an HDP-HMM. Elements are lazily generated.

    Can call where_greater_than(val) to get the indices for which the elements of
    this infinite vector is greater than val.

    Can call __getitem__ to get the element at any index.

    Can call resample(countsdict), where countsdict maps from indices to
    observation counts. Zeros are okay.
    '''
    def __init__(self,alpha_0,beta):
        self.alpha_0 = alpha_0
        self.beta = beta
        self._remaining = 1.
        self._pivec = {}

    def resample(self,countsdict):
        weights = np.random.dirichlet(
                [self.alpha_0*self.beta[k]+n for k,n in countsdict.iteritems()] \
                        + [self.alpha_0*(1-sum(self.beta[k] for k in countsdict))])
        self._pivec = {}
        self._pivec.update(zip(countsdict.keys(),weights[:-1]))
        self._remaining = weights[-1]

    def where_greater_than(self,val):
        while self._remaining > val:
            self[self._unused_index()]
        return [k for k,v in self._pivec.iteritems() if v > val]

    def __getitem__(self,k):
        # not quite stickbreaking, but maintains that for any subset indexed
        # with the symbols {1,2,...,K} we have
        # (pi[1],...,pi[K],pi[rest]) ~ Dir(alpha*beta[1],...,alpha*beta[K],alpha*beta[rest])
        if k not in self._pivec:
            try:
                self._pivec[k], self._remaining = self._remaining * np.random.dirichlet(
                        (self.alpha_0*self.beta[k],
                            self.alpha_0*(1.-self.beta[k]
                                - sum(self.beta[kp] for kp in self._pivec.iterkeys()))))
            except (ZeroDivisionError, FloatingPointError):
                # purely numerical error; floatig point is truncating for us!
                self._pivec[k], self._remaining = self._remaining, 0.
        return self._pivec[k]

    def _unused_index(self):
        unused_globals = set(self.beta._betavec.iterkeys()) - set(self._pivec.iterkeys())
        if len(unused_globals) > 0:
            return unused_globals.pop()
        else:
            # create a new global index
            newidx = len(self.beta._betavec)
            while newidx in self.beta._betavec:
                newidx = np.random.randint(10000)
            return newidx


class HDPHSMMPiRow(HDPHMMPiRow):
    '''
    Represents the pi vectors for an HDP-HSMM. More like \widebar{\pi} in the
    paper notation, since self-transition weights are set to zero in this
    class. Like HDPHMMPiRow except it knows its own index and sets that weight
    to be zero.

    Can call get_aux_counts(num) to get the imagined self-transitions if this
    state was transitioned from NUM times. They are randomly sampled at each
    call.
    '''
    def __init__(self,alpha_0,beta,myidx):
        self.alpha_0 = alpha_0
        self.beta = beta
        self.myidx = myidx
        self._pivec = {myidx:0.}
        self._remaining = 1. # the rest of pi still normalizes

    def resample(self,countsdict):
        assert self.myidx not in countsdict
        weights = np.random.dirichlet(
                [self.alpha_0*self.beta[k]+n for k,n in countsdict.iteritems()] \
                        + [self.alpha_0*(1-sum(self.beta[k] for k in countsdict)
                            - self.beta[self.myidx])]) # NOTE: here's the difference from parent's resample
        self._pivec = {self.myidx:0.}
        self._pivec.update(zip(countsdict.keys(),weights[:-1]))
        self._remaining = weights[-1]

    def get_aux_counts(self,num):
        # sample the latent scaling between pi_ii and the rest of pi
        pi_ii,_ = np.random.dirichlet(
                (self.alpha_0*self.beta[self.myidx],self.alpha_0*(1-self.beta[self.myidx])))
        return np.random.geometric(1-pi_ii,n=num).sum() if num > 0 else 0.


# TODO add counts for initial state, these objects should include pi_0 if
# they're going to encapsulate beta

class HDPHMMBeamTransitions(object):
    def __init__(self,gamma_0,alpha_0):
        self.alpha_0 = alpha_0
        self.beta = Beta(gamma_0)
        self._set_new_pis()

    def _set_new_pis(self):
        self.pis = defaultdict(lambda: HDPHMMPiRow(self.alpha_0,self.beta))

    def resample(self,stateseqlist):
        # count transitions
        counts, idlist = self._count_transitions(stateseqlist)

        # sample m's
        m = np.zeros(counts.shape)
        for (i,j), n in np.ndenumerate(counts):
            m[i,j] = (np.random.randn(n) < self.alpha_0*self.beta[j] \
                    / (np.arange(n) + self.alpha_0*self.beta[j])).sum()
        msum = m.sum(0)

        # resample beta
        self.beta.resample({index:count for index, count in zip(idlist,msum) if count > 0})

        # resample pis
        self._set_new_pis()
        for countrow,ilabel in zip(counts,idlist):
            if countrow.sum() > 0:
                self[ilabel].resample({jlabel:count for jlabel,count in zip(idlist,countrow)
                    if count > 0})

    def _count_transitions(self,stateseqlist):
        id2idx = defaultdict(itertools.count().next)
        numstates = len(reduce(set.union,[set(s) for s in stateseqlist]))
        counts = np.zeros((numstates,numstates))
        for s in stateseqlist:
            for id1,id2 in zip(s[:-1],s[1:]):
                counts[id2idx[id1],id2idx[id2]] += 1
        return counts, map(operator.itemgetter(0),sorted(id2idx.items(),key=operator.itemgetter(1)))

    def __getitem__(self,v):
        if isinstance(v,int):
            return self.pis[v]
        elif isinstance(v,tuple):
            assert len(v) == 2
            return self.pis[v[0]][v[1]]
        else:
            raise IndexError


class HDPHSMMBeamTransitions(HDPHMMBeamTransitions):
    def _set_new_pis(self):
        self.pis = {}

    def _count_transitions(self,stateseqlist):
        stateseqlist = map(operator.itemgetter(0),map(rle,stateseqlist))
        counts, idlist = super(HDPHSMMBeamTransitions,self)._count_transitions(stateseqlist)
        for idx,(label,num) in enumerate(zip(idlist,counts.sum(1))):
            assert counts[idx,idx] == 0
            counts[idx,idx] = self[idlist].get_aux_counts(num)
        return counts, idlist

    def __getitem__(self,v):
        if isinstance(v,int):
            if v not in self.pis:
                self.pis[v] = HDPHSMMPiRow(self.alpha_0,self.beta,v)
            return self.pis[v]
        elif isinstance(v,tuple):
            assert len(v) == 2
            return self[v[0]][v[1]]
        else:
            raise IndexError

