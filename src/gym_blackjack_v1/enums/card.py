#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum


class Card(IntEnum):
    _2 = 0
    _3 = 1
    _4 = 2
    _5 = 3
    _6 = 4
    _7 = 5
    _8 = 6
    _9 = 7
    _T = 8 # 10, J, Q, K are all denoted as T
    _A = 9


card_labels = [ c.name[1:] for c in Card ]
