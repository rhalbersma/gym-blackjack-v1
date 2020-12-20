#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

from ..enums import Card, Hand, Player

################################################################################
# Factor all transition logic into a finite-state machine (FSM) lookup table.
# Note that this is model-free RL because there are no explicit probabilities.
#
# The FSM approach has two main benefits:
# 1) it speeds up the model-free version of the BlackjackEnv class;
# 2) it is straightforward to extend to a model-based version.
################################################################################

# Going from a state to the next state after 'hitting' another card.
hit = np.zeros((len(Player), len(Card)), dtype=int)

for _j, _c in enumerate(range(Card._2, Card._T)):
    hit[Player.NONE, _c] = Player.H2 + _j
hit[Player.NONE, Card._T] = Player.T
hit[Player.NONE, Card._A] = Player.A

for _i, _h in enumerate(range(Player.H2, Player.H11)):
    for _j, _c in enumerate(range(Card._2, Card._A)):
        hit[_h, _c] = _h + 2 + _j
    hit[_h, Card._A] = Player.S13 + _i

for _j, _c in enumerate(range(Card._2, Card._A)):
    hit[Player.H11, _c] = Player.H13 + _j
hit[Player.H11, Card._A] = Player.H12

for _i, _h in enumerate(range(Player.H12, Player.H21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        hit[_h, _c] = _h + 2 + _j
    hit[_h, (Card._T - _i):Card._A] = Player._BUST
    hit[_h, Card._A] = _h + 1

hit[Player.H21, :] = Player._BUST

for _j, _c in enumerate(range(Card._2, Card._A)):
    hit[Player.T, _c] = Player.H12 + _j
hit[Player.T, Card._A] = Player.BJ

for _j, _c in enumerate(range(Card._2, Card._T)):
    hit[Player.A, _c] = Player.S13 + _j
hit[Player.A, Card._T] = Player.BJ
hit[Player.A, Card._A] = Player.S12

for _i, _s in enumerate(range(Player.S12, Player.S21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        hit[_s, _c] = _s + 2 + _j
    for _j, _c in enumerate(range(Card._T - _i, Card._A)):
        hit[_s, _c] = Player.H12 + _j
    hit[_s, Card._A] = _s + 1

for _c, _j in enumerate(range(Card._2, Card._A)):
    hit[Player.S21, _c] = Player.H13 + _j
hit[Player.S21, Card._A] = Player.H12

hit[Player.BJ, :] = hit[Player.S21, :]

hit[Player._BUST:, :] = Player._END

# Going from one state to the next state after 'standing'.
stand = np.zeros(len(Player), dtype=int)

stand[:Player.H17] = Player._16

for _i, _h in enumerate(range(Player.H17, Player.H21 + 1)):
    stand[_h] = Player._17 + _i
stand[Player.T:Player.S17] = Player._16

for _i, _s in enumerate(range(Player.S17, Player.S21 + 1)):
    stand[_s] = Player._17 + _i
stand[Player.BJ] = Player._BJ

stand[Player._BUST:] = Player._END

# Going from a Hand to a Count.
count = np.zeros(len(Player), dtype=int)

count[:len(Hand)] = stand[:len(Hand)] - len(Hand)

for _s in range(Player._BUST, Player._BJ + 1):
    count[_s] = _s - len(Hand)

count[Player._END] = count[Player._BUST]
