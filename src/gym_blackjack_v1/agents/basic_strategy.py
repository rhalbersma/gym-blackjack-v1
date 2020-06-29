#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
from ..envs import State, Card, Action

class BasicStrategyAgent:
    """
    A blackjack agent that plays according to the Basic Strategy (Thorp, 1966).
    """
    def __init__(self, env):
        self.policy = np.full((len(State), len(Card)), Action.h)
        self.policy[State.H13:(State.H21 + 1), Card._2:(Card._3 + 1)] = Action.s
        self.policy[State.H12:(State.H21 + 1), Card._4:(Card._6 + 1)] = Action.s
        self.policy[State.H17:(State.H21 + 1), Card._7:(Card._A + 1)] = Action.s
        self.policy[State.S18:(State.BJ  + 1), Card._2:(Card._8 + 1)] = Action.s
        self.policy[State.S19:(State.BJ  + 1), Card._9:(Card._A + 1)] = Action.s
        self.policy[State.BUST,                :                    ] = Action.s

        self.observe = {
            'Blackjack-v0': (lambda obs: (State[('S' if obs[2] else 'H') + str(obs[0])], Card((obs[1] - 2) % 10))),
            'Blackjack-v1': (lambda obs: obs)
        }[env.unwrapped.spec.id]

    def act(self, obs, reward, done):
        return self.policy[self.observe(obs)]
