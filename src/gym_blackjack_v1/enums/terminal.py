#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum


class Terminal(IntEnum):
    _END = 0


terminal_labels = [ t.name[1:] for t in Terminal ]
