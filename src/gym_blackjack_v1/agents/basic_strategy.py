#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

from ..enums import Action, Card, Hand, nC, nH


class BasicStrategyAgent:
    """
    A blackjack agent that plays according to the Basic Strategy of Thorp (1966).
    """
    def __init__(self, env):
        self.policy = np.full((nH, nC), Action.HIT)
        self.policy[Hand.H13:(Hand.H21 + 1), Card._2:(Card._3 + 1)] = Action.STAND
        self.policy[Hand.H12:(Hand.H21 + 1), Card._4:(Card._6 + 1)] = Action.STAND
        self.policy[Hand.H17:(Hand.H21 + 1), Card._7:(Card._A + 1)] = Action.STAND
        self.policy[Hand.S18:(Hand.BJ  + 1), Card._2:(Card._8 + 1)] = Action.STAND
        self.policy[Hand.S19:(Hand.BJ  + 1), Card._9:(Card._A + 1)] = Action.STAND

        self.decode = {
            'Blackjack-v0': (lambda obs: (Hand[('S' if obs[2] else 'H') + str(obs[0])], Card((obs[1] - 2) % nC))),
            'Blackjack-v1': (lambda obs: divmod(obs, nC))
        }[env.unwrapped.spec.id]

    def act(self, obs, reward, done):
        return self.policy[self.decode(obs)]

