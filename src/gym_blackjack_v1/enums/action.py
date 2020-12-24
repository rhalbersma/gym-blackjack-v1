#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum


class Action(IntEnum):
    STAND = 0
    HIT   = 1


action_labels = [ a.name[0] for a in Action ]
