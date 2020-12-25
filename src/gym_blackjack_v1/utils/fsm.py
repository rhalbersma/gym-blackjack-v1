#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

from ..enums import Card, State

################################################################################
# Factor all transition logic into a finite-state machine (FSM) lookup table.
# Note that this is model-free RL because there are no explicit probabilities.
#
# The FSM approach has two main benefits:
# 1) it is fast to simulate the model-free version of the BlackjackEnv class;
# 2) combining the FSM with card probabilities leads to a model-based version.
################################################################################

# Going from a State to the next State after 'hitting' another Card.
hit = np.full((len(State), len(Card)), -1, dtype=int)

hit[State._DEAL, Card._2] = State._DEUCE
hit[State._DEAL, Card._3] = State._TREY
hit[State._DEAL, Card._T] = State._TEN
hit[State._DEAL, Card._A] = State._ACE
for _j, _c in enumerate(range(Card._4, Card._T)):
    hit[State._DEAL, _c] = State.H4 + _j

for _i, _h in enumerate((State._DEUCE, State._TREY)):
    for _j, _c in enumerate(range(Card._2, Card._A)):
        hit[_h, _c] = State.H4 + _i + _j
    hit[_h, Card._A] = State.S13 + _i

for _j, _c in enumerate(range(Card._2, Card._A)):
    hit[State._TEN, _c] = State.H12 + _j
hit[State._TEN, Card._A] = State.BJ

for _j, _c in enumerate(range(Card._2, Card._T)):
    hit[State._ACE, _c] = State.S13 + _j
hit[State._ACE, Card._T] = State.BJ
hit[State._ACE, Card._A] = State.S12

for _i, _h in enumerate(range(State.H4, State.H11)):
    for _j, _c in enumerate(range(Card._2, Card._A)):
        hit[_h, _c] = State.H6 + _i + _j
    hit[_h, Card._A] = State.S15 + _i

for _j, _c in enumerate(range(Card._2, Card._A)):
    hit[State.H11, _c] = State.H13 + _j
hit[State.H11, Card._A] = State.H12

for _i, _h in enumerate(range(State.H12, State.H21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        hit[_h, _c] = State.H14 + _i + _j
    hit[_h, (Card._T - _i):Card._A] = State._BUST
    hit[_h, Card._A] = State.H13 + _i

hit[State.H21, :] = State._BUST

for _i, _s in enumerate(range(State.S12, State.S21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        hit[_s, _c] = State.S14 + _i + _j
    for _j, _c in enumerate(range(Card._T - _i, Card._A)):
        hit[_s, _c] = State.H12 +  _j
    hit[_s, Card._A] = State.S13 + _i

for _c, _j in enumerate(range(Card._2, Card._A)):
    hit[State.S21, _c] = State.H13 + _j
hit[State.S21, Card._A] = State.H12

hit[State.BJ, :] = hit[State.S21, :]

for _c in range(State._BUST, State._BJ + 1):
    hit[_c] = _c

# Going from one State to the next State after 'standing'.
stand = np.full(len(State), State._16, dtype=int)

for _i, _h in enumerate(range(State.H17, State.H21 + 1)):
    stand[_h] = State._17 + _i

for _i, _s in enumerate(range(State.S17, State.S21 + 1)):
    stand[_s] = State._17 + _i

stand[State.BJ] = State._BJ

for _c in range(State._BUST, State._BJ + 1):
    stand[_c] = _c

# Combined FSM for model-building
stand_hit = np.array([a for a in np.broadcast_arrays(stand.reshape(-1, 1), hit)])

# Going from a State to a Count.
count = stand - State._BUST
