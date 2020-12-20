#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum, extend_enum

from .count import Count, count_labels
from .hand import Hand, hand_labels
from .terminal import Terminal, terminal_labels


class Player(IntEnum):
    pass


for key, value in Hand.__members__.items():
    extend_enum(Player, key, value)
for key, value in Count.__members__.items():
    extend_enum(Player, key, value + len(Hand))
for key, value in Terminal.__members__.items():
    extend_enum(Player, key, value + len(Hand) + len(Count))

player_labels = hand_labels + count_labels + terminal_labels
