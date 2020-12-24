from aenum import IntEnum, extend_enum

from .count import Count, count_labels
from .hand import Hand, hand_labels
from .single import Single, single_labels


class State(IntEnum):
    pass


for key, value in Hand.__members__.items():
    extend_enum(State, key, value)
for key, value in Single.__members__.items():
    extend_enum(State, key, value + len(Hand))
for key, value in Count.__members__.items():
    extend_enum(State, key, value + len(Hand) + len(Single))

state_labels = hand_labels + single_labels + count_labels
