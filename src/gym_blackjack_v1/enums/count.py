#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum


class Count(IntEnum):
    _BUST =  0 # all counts above 21
    _16   =  1 # all counts below 17
    _17   =  2
    _18   =  3
    _19   =  4
    _20   =  5
    _21   =  6
    _BJ   =  7 # 21 with the first 2 cards


count_labels = [ c.name[1:] for c in Count ]
