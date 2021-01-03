#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum


class Hand(IntEnum):
    H4   =  0
    H5   =  1
    H6   =  2
    H7   =  3
    H8   =  4
    H9   =  5
    H10  =  6
    H11  =  7
    H12  =  8
    H13  =  9
    H14  = 10
    H15  = 11
    H16  = 12
    H17  = 13
    H18  = 14
    H19  = 15
    H20  = 16
    H21  = 17
    S12  = 18
    S13  = 19
    S14  = 20
    S15  = 21
    S16  = 22
    S17  = 23
    S18  = 24
    S19  = 25
    S20  = 26
    S21  = 27
    BJ   = 28


hand_labels = [ h.name for h in Hand ]

nH = len(Hand)
