from aenum import IntEnum, extend_enum

from .hand import Hand, hand_labels, nH
from .single import Single, single_labels
from .terminal import Terminal, terminal_labels


class Markov(IntEnum):
    pass


for key, value in Hand.__members__.items():
    extend_enum(Markov, key, value)
for key, value in Single.__members__.items():
    extend_enum(Markov, key, value + nH)
for key, value in Terminal.__members__.items():
    extend_enum(Markov, key, value + nH + len(Single))

markov_labels = hand_labels + single_labels + terminal_labels

nM = len(Markov)
