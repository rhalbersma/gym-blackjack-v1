#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

from ..enums import Card


class InfiniteDeck:
    """
    An infinite deck of cards (i.e. drawing from a single suit with replacement).
    """
    def __init__(self, np_random):
        self.cards = np.array([ c for c in range(Card._2, Card._T) ] + [ Card._T ] * 4 + [ Card._A ])
        self.np_random = np_random
        _, counts = np.unique(self.cards, return_counts=True)
        self.prob = counts / counts.sum()
        assert np.isclose(self.prob.sum(), 1)

    def draw(self):
        """
        Draw a single card.
        """
        return self.np_random.choice(self.cards, replace=True)

    def deal(self):
        """
        Draw two player cards and one dealer card.
        """
        return self.draw(), self.draw(), self.draw()

