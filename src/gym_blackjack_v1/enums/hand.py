#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum


class Hand(IntEnum):
    NONE =  0
    H2   =  1
    H3   =  2
    H4   =  3
    H5   =  4
    H6   =  5
    H7   =  6
    H8   =  7
    H9   =  8
    H10  =  9
    H11  = 10
    H12  = 11
    H13  = 12
    H14  = 13
    H15  = 14
    H16  = 15
    H17  = 16
    H18  = 17
    H19  = 18
    H20  = 19
    H21  = 20
    T    = 21
    A    = 22
    S12  = 23
    S13  = 24
    S14  = 25
    S15  = 26
    S16  = 27
    S17  = 28
    S18  = 29
    S19  = 30
    S20  = 31
    S21  = 32
    BJ   = 33


hand_labels = [ h.name for h in Hand ]
