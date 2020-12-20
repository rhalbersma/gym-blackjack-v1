#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

from ..enums import Action, Hand

# Deterministic policy for a dealer who stands on 17.
stands_on_17 = np.full(len(Hand), Action.h)

for _h in range(Hand.H17, Hand.H21 + 1):
    stands_on_17[_h] = Action.s

for _s in range(Hand.S17, Hand.BJ + 1):
    stands_on_17[_s] = Action.s

# Deterministic policy for a dealer who hits on soft 17.
hits_on_soft_17 = stands_on_17.copy()
hits_on_soft_17[Hand.S17] = Action.h
