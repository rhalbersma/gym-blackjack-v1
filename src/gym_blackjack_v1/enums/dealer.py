#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum, extend_enum

from .card import Card, card_labels
from .terminal import Terminal, terminal_labels


class Dealer(IntEnum):
    pass


for key, value in Card.__members__.items():
    extend_enum(Dealer, key, value)
for key, value in Terminal.__members__.items():
    extend_enum(Dealer, key, value + len(Card))

dealer_labels = card_labels + terminal_labels
