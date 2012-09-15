from __future__ import division
import numpy as np

from pyhsmm.util.general import irle
from pyhsmm.util.stats import sample_discrete_from_log

class HMMBeamStates(object):
    def __init__(self,A,obs_distns,stateseq=None,T=None):
        self.A = A
        self.pi_0 = self.A.pi_0
        self.obs_distns = obs_distns

        if stateseq is None:
            assert T is not None
            self.generate_states(T)
        else:
            self.stateseq = stateseq

    def generate_states(self,T):
        stateseq = np.empty(T)
        nextstate_distn = self.pi_0

        for t in range(T):
            stateseq[t] = nextstate_distn.rvs(1)
            nextstate_distn = self.A[stateseq[t]]

        self.stateseq = stateseq

    def resample(self):
        us = self._sample_useq()
        possible_states = self._possible_states(us)
        Als, aBls, betals = self._messages_backwards(us,possible_states)
        self._sample_forwards(Als,aBls,betals,possible_states)

    def _sample_useq(self):
        us = np.empty(self.T)
        us[0] = np.random.uniform(low=0.,high=self.pi_0[self.stateseq[0]])
        for t,state in enumerate(self.stateseq[1:]):
            us[t+1] = np.random.uniform(low=0.,high=self.A[self.stateseq[t]])
        return us

    def _possible_states(self,us):
        possible_states = [self.pi_0.where_greate_than(us[0])]
        for u in us[1:]:
            possible_states.append(list(reduce(set.union,
                set([self.A[k].where_greater_than(u) for k in possible_states[-1]]))))
        return possible_states

    def _messages_backwards(self,us,possible_states):
        # first, set up lists of arrays
        Als, aBls = [], []
        for l1,l2,d in zip(possible_states[:-1],possible_states[1:],self.data[1:]):
            Als.append(np.log(np.array([[self.A[krow,kcol] for kcol in l2] for krow in l1])))
            aBls.append(np.array([self.obs_distns[k].log_likelihood(d) for k in l2]))

        # now a backwards pass using fast matrix operations
        betals = []
        for Al, aBl in reversed(zip(Als,aBls)):
            betals.insert(0,np.logaddexp.reduce(Al + betals[0] + aBl,axis=1))

        # add the first evidence term into aBls
        aBls.insert(0,np.array([self.obs_distns[k].log_likelihood(self.data[0]) for k in possible_states[0]]))

        assert len(Als)+1 == len(aBls) == len(betals)+1 == self.T

        return Als, aBls, betals

    def _sample_forwards(self,Als,aBls,betals,possible_states):
        stateseq = np.empty(self.T,dtype=np.int)

        log_nextstate_unsmoothed = np.log(np.array([self.pi_0[k] for k in possible_states[0]]))
        for t,(labels,aBl,betal) in enumerate(zip(possible_states[:-1],aBls[:-1],betals[:-1])):
            stateseq[t] = labels[sample_discrete_from_log(aBl + betal + log_nextstate_unsmoothed)]
            log_nextstate_unsmoothed = np.log(np.array([self.A[stateseq[t],k] for k in possible_states[t+1]]))
        stateseq[-1] = possible_states[-1][sample_discrete_from_log(aBls[-1] + log_nextstate_unsmoothed)]

        self.stateseq = stateseq


class HSMMBeamStates(HMMBeamStates):
    def __init__(self):
        raise NotImplementedError

    def generate_states(self):
        raise NotImplementedError

    def resample(self):
        us = self._sample_useq()
        possible_states = self._possible_states(us)
        Als, aBls, betals, betastarls = self._messages_backwards(us,possible_states)
        self._sample_forwards(Als,aBls,betals,betastarls,possible_states)

    def _sample_useq(self):
        u = np.empty(self.T)
        u[0] = np.random.uniform(low=0.,high=self.pi_0[self.stateseq[0]])
        for t,state in enumerate(self.stateseq[1:]):
            u[t+1] = np.random.uniform(low=0.,high=self.A[self.stateseq[t]])
        return u

    def _possible_states(us):
        raise NotImplementedError

    def _messages_backwards(self):
        raise NotImplementedError

    def _sample_frwards(self):
        raise NotImplementedError

