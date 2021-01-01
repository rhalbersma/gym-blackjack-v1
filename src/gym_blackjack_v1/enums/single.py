    #          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum


class Single(IntEnum):
    _DEAL  = 0
    _DEUCE = 1
    _TREY  = 2
    _TEN   = 3  # 10, J, Q, K are all denoted as TEN
    _ACE   = 4


single_labels = [ s.name[1:] for s in Single ]
