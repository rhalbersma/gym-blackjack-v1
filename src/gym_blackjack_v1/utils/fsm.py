#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

from ..enums import Card, Markov, nC, nM

################################################################################
# Factor all transition logic into a finite-state machine (FSM) lookup table.
# Note that this is model-free RL because there are no explicit probabilities.
#
# The FSM approach has two main benefits:
# 1) it is fast to simulate the model-free version of the BlackjackEnv class;
# 2) combining the FSM with card probabilities leads to a model-based version.
################################################################################

# Going from a Markov state to the next after 'hitting' another Card.
hit = np.full((nM, nC), -1, dtype=int)

hit[Markov._DEAL, Card._2] = Markov._DEUCE
hit[Markov._DEAL, Card._3] = Markov._TREY
hit[Markov._DEAL, Card._T] = Markov._TEN
hit[Markov._DEAL, Card._A] = Markov._ACE
for _j, _c in enumerate(range(Card._4, Card._T)):
    hit[Markov._DEAL, _c] = Markov.H4 + _j

for _i, _h in enumerate((Markov._DEUCE, Markov._TREY)):
    for _j, _c in enumerate(range(Card._2, Card._A)):
        hit[_h, _c] = Markov.H4 + _i + _j
    hit[_h, Card._A] = Markov.S13 + _i

for _j, _c in enumerate(range(Card._2, Card._A)):
    hit[Markov._TEN, _c] = Markov.H12 + _j
hit[Markov._TEN, Card._A] = Markov.BJ

for _j, _c in enumerate(range(Card._2, Card._T)):
    hit[Markov._ACE, _c] = Markov.S13 + _j
hit[Markov._ACE, Card._T] = Markov.BJ
hit[Markov._ACE, Card._A] = Markov.S12

for _i, _h in enumerate(range(Markov.H4, Markov.H11)):
    for _j, _c in enumerate(range(Card._2, Card._A)):
        hit[_h, _c] = Markov.H6 + _i + _j
    hit[_h, Card._A] = Markov.S15 + _i

for _j, _c in enumerate(range(Card._2, Card._A)):
    hit[Markov.H11, _c] = Markov.H13 + _j
hit[Markov.H11, Card._A] = Markov.H12

for _i, _h in enumerate(range(Markov.H12, Markov.H21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        hit[_h, _c] = Markov.H14 + _i + _j
    hit[_h, (Card._T - _i):Card._A] = Markov._BUST
    hit[_h, Card._A] = Markov.H13 + _i

hit[Markov.H21, :] = Markov._BUST

for _i, _s in enumerate(range(Markov.S12, Markov.S21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        hit[_s, _c] = Markov.S14 + _i + _j
    for _j, _c in enumerate(range(Card._T - _i, Card._A)):
        hit[_s, _c] = Markov.H12 +  _j
    hit[_s, Card._A] = Markov.S13 + _i

for _c, _j in enumerate(range(Card._2, Card._A)):
    hit[Markov.S21, _c] = Markov.H13 + _j
hit[Markov.S21, Card._A] = Markov.H12

hit[Markov.BJ, :] = hit[Markov.S21, :]

for _c in range(Markov._BUST, Markov._BJ + 1):
    hit[_c] = _c

# Going from one Markov state to the next after 'standing'.
stand = np.full(nM, Markov._16, dtype=int)

for _i, _h in enumerate(range(Markov.H17, Markov.H21 + 1)):
    stand[_h] = Markov._17 + _i

for _i, _s in enumerate(range(Markov.S17, Markov.S21 + 1)):
    stand[_s] = Markov._17 + _i

stand[Markov.BJ] = Markov._BJ

for _c in range(Markov._BUST, Markov._BJ + 1):
    stand[_c] = _c

# Combined FSM for model-building
stand_hit = np.array([a for a in np.broadcast_arrays(stand.reshape(-1, 1), hit)])

# Going from a Markov state to a Count.
count = stand - Markov._BUST
