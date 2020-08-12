#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
from ..envs import Hand, Card, Action

class StandOn20Agent:
    """
    A blackjack agent that stands on 20 or higher (Sutton and Barto, 2018).
    """
    def __init__(self, env):
        self.policy = np.full((len(Hand), len(Card)), Action.h)
        self.policy[Hand.H20:(Hand.H21 + 1), :] = Action.s
        self.policy[Hand.S20:(Hand.BJ  + 1), :] = Action.s

        self.observe = {
            'Blackjack-v0': (lambda obs: (Hand[('S' if obs[2] else 'H') + str(obs[0])], Card((obs[1] - 2) % 10))),
            'Blackjack-v1': (lambda obs: obs)
        }[env.unwrapped.spec.id]

    def act(self, obs, reward, done):
        return self.policy[self.observe(obs)]