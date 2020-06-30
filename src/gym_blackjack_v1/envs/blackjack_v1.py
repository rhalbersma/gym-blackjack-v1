#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from enum import IntEnum
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Action(IntEnum):
    s = 0
    h = 1

action_labels = [ a.name[0] for a in Action ]

class Card(IntEnum):
    _2 =  0
    _3 =  1
    _4 =  2
    _5 =  3
    _6 =  4
    _7 =  5
    _8 =  6
    _9 =  7
    _T =  8
    _A =  9

card_labels = [ c.name[1:] for c in Card ]

# Drawing cards from an infinite deck (i.e. from a single deck with replacement)
deck = np.array([ c for c in range(Card._2, Card._T) ] + [ Card._T ] * 4 + [ Card._A ])

class State(IntEnum):
    DEAL =  0
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
    BUST = 34

state_labels = [ s.name for s in State ]

# Finite-state machine for going from one state to the next after drawing a card
fsm = np.zeros((len(State), len(Card)), dtype=int)

for _j, _c in enumerate(range(Card._2, Card._T)):
    fsm[State.DEAL, _c] = State.H2 + _j
fsm[State.DEAL, Card._T] = State.T
fsm[State.DEAL, Card._A] = State.A

for _i, _h in enumerate(range(State.H2, State.H11)):
    for _j, _c in enumerate(range(Card._2, Card._A)):
        fsm[_h, _c] = _h + 2 + _j
    fsm[_h, Card._A] = State.S13 + _i

for _j, _c in enumerate(range(Card._2, Card._A)):
    fsm[State.H11, _c] = State.H13 + _j
fsm[State.H11, Card._A] = State.H12

for _i, _h in enumerate(range(State.H12, State.H21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        fsm[_h, _c] = _h + 2 + _j
    fsm[_h, (Card._T - _i):Card._A] = State.BUST
    fsm[_h, Card._A] = _h + 1

fsm[State.H21, :] = State.BUST

for _j, _c in enumerate(range(Card._2, Card._A)):
    fsm[State.T, _c] = State.H12 + _j
fsm[State.T, Card._A] = State.BJ

for _j, _c in enumerate(range(Card._2, Card._T)):
    fsm[State.A, _c] = State.S13 + _j
fsm[State.A, Card._T] = State.BJ
fsm[State.A, Card._A] = State.S12

for _i, _s in enumerate(range(State.S12, State.S21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        fsm[_s, _c] = _s + 2 + _j
    for _j, _c in enumerate(range(Card._T - _i, Card._A)):
        fsm[_s, _c] = State.H12 + _j
    fsm[_s, Card._A] = _s + 1

for _c, _j in enumerate(range(Card._2, Card._A)):
    fsm[State.S21, _c] = State.H13 + _j
fsm[State.S21, Card._A] = State.H12

fsm[State.BJ,   :] = fsm[State.S21, :]
fsm[State.BUST, :] = State.BUST

# Finite-state machine for a dealer who has to hit on soft 17
hit_on_soft_17 = fsm.copy()

for _h in range(State.H17, State.H21 + 1):
    hit_on_soft_17[_h, :] = _h

for _s in range(State.S18, State.BJ + 1):
    hit_on_soft_17[_s, :] = _s

# Finite-state machine for a dealer who has to stand on 17
stand_on_17 = hit_on_soft_17.copy()
stand_on_17[State.S17, :] = State.S17

# Terminal states
class Terminal(IntEnum):
    _BUST =  0 # all counts above 21
    _16   =  1 # all counts below 17
    _17   =  2
    _18   =  3
    _19   =  4
    _20   =  5
    _21   =  6
    _BJ   =  7 # 21 with the first 2 cards

# Terminal state labels
terminals = [ t.name[1:] for t in Terminal ]

# Transition matrix to terminal states when standing
score = np.zeros(len(State), dtype=int)

score[State.BUST          ] = Terminal._BUST
score[State.DEAL:State.H17] = Terminal._16
score[State.T   :State.S17] = Terminal._16

for _i, _h in enumerate(range(State.H17, State.H21 + 1)):
    score[_h] = Terminal._17 + _i

for _i, _s in enumerate(range(State.S17, State.S21 + 1)):
    score[_s] = Terminal._17 + _i

score[State.BJ] = Terminal._BJ

# The payout structure as specified in Sutton and Barto.
_sutton_barto = np.zeros((len(Terminal), len(Terminal)))    # The player and dealer have equal scores.
_sutton_barto[Terminal._BUST, :             ]       = -1.   # The player busts regardless of whether the dealer busts.
_sutton_barto[Terminal._16: , Terminal._BUST]       = +1.   # The dealer busts and the player doesn't.
_sutton_barto[np.tril_indices(len(Terminal), k=-1)] = +1.   # The player scores higher than the dealer.
_sutton_barto[np.triu_indices(len(Terminal), k=+1)] = -1.   # The dealer scores higher than the player.

# The payout structure as specified in the default Blackjack-v0 environment with natural=False.
# Note: in contrast to Sutton and Barto, blackjack is considered equivalent to 21.
_blackjack_v0 = _sutton_barto.copy()
_blackjack_v0[Terminal._BJ, Terminal._21]           =  0.   # A player's blackjack and a dealer's 21 are treated equally.
_blackjack_v0[Terminal._21, Terminal._BJ]           =  0.   # A player's 21 and a dealer's blackjack are treated equally.

# The payout structure as specified in the alternative Blackjack-v0 environment with natural=True.
# Note: in contrast to Sutton and Barto, blackjack is considered equivalent to 21.
_blackjack_v0_natural = _blackjack_v0.copy()
_blackjack_v0_natural[Terminal._BJ, :Terminal._21]   = +1.5 # A player's winning blackjack pays 1.5 times the original bet.

# The typical casino payout structure as specified in E.O. Thorp, "Beat the Dealer" (1966).
# https://www.amazon.com/gp/product/B004G5ZTZQ/
_thorp = _sutton_barto.copy()
_thorp[Terminal._BJ, :Terminal._BJ]                  = +1.5 # A player's winning blackjack pays 1.5 times the original bet.

class BlackjackEnv(gym.Env):
    """
    Blackjack environment corresponding to Examples 5.1, 5.3 and 5.4 in
    Reinforcement Learning: An Introduction (2nd ed.) by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html

    Description:
        The object of the popular casino card game of blackjack is to obtain cards,
        the sum of whose values is as great as possible without exceeding 21 ('bust').
        Face cards count as 10, and an ace can count as either 1 ('hard') or 11 ('soft').

    Observations:
        There are 35 * 10 = 350 discrete states:
            35 player counts (DEAL, H2-H21, TEN, ACE, S12-S21, BJ, BUST)
            10 dealer cards showing (2-9, TEN, ACE)

    Actions:
        There are 2 actions for the player:
            0: stop the game ('stand' or 'stick') and play out the dealer's hand.
            1: request an additional card to be dealt ('hit').
        Note: additional actions allowed in casino play, such as taking insurance,
        doubling down, splitting or surrendering, are not supported.

    Rewards:
        There are 3 rewards for the player:
            -1. : the player busted, regardless of the dealer,
                or the player's total is lower than the dealer's,
                or the dealer has a blackjack and the player doesn't.
             0. : the player's total is equal to the dealer's.
            +1. : the player's total is higher than the dealer's,
                or the dealer busted and the player didn't,
                or the player has a blackjack and the dealer doesn't
                (in casino play, this pays out +1.5 to the player).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, payout=None, dealer_hits_on_soft_17=False):
        """
        Initialize the state of the environment.

        Args:
            payout (None, str or an 8x8 NumPy matrix): the game's payout structure.
                If payout is None or 'sutton-barto', blackjack beats 21 and a player's winning blackjack pays out +1.0.
                If payout is 'blackjack-v0', same as above, but a blackjack ties with 21.
                If payout is 'blackjack-v0-natural', same as above, but a player's winning blackjack pays out +1.5.
                If payout is 'thorp', same as 'sutton-barto', but a player's winning blackjack pays out +1.5.
            dealer_hits_on_soft (bool): whether the dealer stands or hits on soft 17.
            debug (bool): whether the environment keeps track of the cards received by the player and the dealer.

        Notes:
            'suttton-barto' is described in Sutton and Barto's "Reinforcement Learning" (2018).
            'blackjack-v0' is implemented in the OpenAI Gym environment 'Blackjack-v0'.
            'blackjack-v0-natural' is obtained from 'Blackjack-v0' initialized with natural=True.
            'thorp' is described in E.O. Thorp's "Beat the Dealer" (1966).
        """
        if payout is None:
            self.payout = _sutton_barto
        elif isinstance(payout, np.ndarray) and payout.shape == (len(Terminal), len(Terminal)) and payout.dtype.name.startswith('float'):
            self.payout = payout
        elif isinstance(payout, str):
            try:
                self.payout = {
                    'sutton-barto'          : _sutton_barto,
                    'blackjack-v0'          : _blackjack_v0,
                    'blackjack-v0-natural'  : _blackjack_v0_natural,
                    'thorp'                 : _thorp
                }[payout.lower()]   # Handle mixed-case spelling.
            except KeyError:
                raise ValueError(f"Unknown payout name '{payout}'")
        else:
            raise ValueError(f"Unknown payout type '{type(payout)}'")
        terminal_states = lambda fsm: {
            state
            for state, successors in enumerate(fsm)
            if (successors == state).all()
        }
        self.player_fsm = fsm
        self.player_terminal = terminal_states(self.player_fsm)
        self.dealer_fsm = hit_on_soft_17 if dealer_hits_on_soft_17 else stand_on_17
        self.dealer_terminal = terminal_states(self.dealer_fsm)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(len(State)),
            spaces.Discrete(len(Card))
        ))
        self.action_space = spaces.Discrete(len(Action))
        self.reward_range = (np.min(self.payout), np.max(self.payout))
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        p = states[self.player]
        upcard_only = self.dealer in range(Card._2, Card._A + 1)
        if self.player not in self.player_terminal and upcard_only:
            d = cards[self.dealer]
            return f'player: {p:>4}; dealer: {d:>4};'
        else:
            d = states[self.dealer_fsm[State.DEAL, self.dealer] if upcard_only else self.dealer]
            R = self.payout[score[self.player], score[self.dealer]]
            return f'player: {p:>4}; dealer: {d:>4}; reward: {R:>+4}'

    def _draw(self):
        return self.np_random.choice(deck, replace=True)

    def _deal(self):
        p1, p2, up = self._draw(), self._draw(), self._draw()
        self.info = { 'player': [ cards[p1], cards[p2] ], 'dealer': [ cards[up] ] }
        return self.player_fsm[self.player_fsm[State.DEAL, p1], p2], up

    def _get_obs(self):
        return self.player, self.dealer

    def reset(self):
        """
        Reset the state of the environment by drawing 2 player cards and 1 dealer card.

        Returns:
            observation (object): the player's count and the dealer's upcard.

        Notes:
            Cards are drawn from a single deck with replacement (i.e. from an infinite deck).
        """
        self.player, self.dealer = self._deal()
        return self._get_obs()

    def explore(self, start):
        """
        Explore a specific starting state of the environment.

        Args:
            start (tuple of ints): the player's count and the dealer's upcard.

        Returns:
            observation (object): the player's count and the dealer's upcard.

        Notes:
            Monte Carlo Exploring Starts should use this method instead of reset().
        """
        self.player, self.dealer = start
        self.info = { 'player': [], 'dealer': [] }
        return self._get_obs()

    def step(self, action):
        if action == Action.h:
            next = self._draw()
            self.info['player'].append(cards[next])
            self.player = self.player_fsm[self.player, next]
            done = self.player in self.player_terminal
        else:
            # Monte Carlo Exploring Starts will periodically sample the player's bust state.
            # In that case, the dealer should not draw additional cards that could bust him.
            if self.player not in self.player_terminal:
                self.dealer = self.dealer_fsm[State.DEAL, self.dealer]
                while True:
                    next = self._draw()
                    self.info['dealer'].append(cards[next])
                    self.dealer = self.dealer_fsm[self.dealer, next]
                    if self.dealer in self.dealer_terminal:
                        break
            done = True
        reward = self.payout[score[self.player], score[self.dealer]] if done else 0.
        return self._get_obs(), reward, done, self.info
