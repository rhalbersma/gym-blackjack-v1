#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum, extend_enum
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Hand(IntEnum):
    """
    All 'live' hands in blackjack (transient Markov states).
    """
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

hand_labels = [ h.name for h in Hand ]

class Count(IntEnum):
    """
    All 'dead' hands in blackjack (absorbing Markov states).
    """
    _BUST =  0 # all counts above 21
    _16   =  1 # all counts below 17
    _17   =  2
    _18   =  3
    _19   =  4
    _20   =  5
    _21   =  6
    _BJ   =  7 # 21 with the first 2 cards

count_labels = [ c.name[1:] for c in Count ]

class State(IntEnum):
    """
    The union of all 'live' and 'dead' hands in blackjack.
    """
    pass

for key, value in Hand.__members__.items():
    extend_enum(State, key, value)

for key, value in Count.__members__.items():
    extend_enum(State, key, value + len(Hand))

state_labels = hand_labels + count_labels

class Card(IntEnum):
    _2 =  0
    _3 =  1
    _4 =  2
    _5 =  3
    _6 =  4
    _7 =  5
    _8 =  6
    _9 =  7
    _T =  8 # 10, J, Q, K are all denoted as T
    _A =  9

card_labels = [ c.name[1:] for c in Card ]

class Action(IntEnum):
    s = 0
    h = 1

action_labels = [ a.name[0] for a in Action ]

# Finite-state machine for going from a hand to the next state after 'hitting' another card.
fsm_hit = np.zeros((len(Hand), len(Card)), dtype=int)

for _j, _c in enumerate(range(Card._2, Card._T)):
    fsm_hit[Hand.DEAL, _c] = State.H2 + _j
fsm_hit[Hand.DEAL, Card._T] = State.T
fsm_hit[Hand.DEAL, Card._A] = State.A

for _i, _h in enumerate(range(Hand.H2, Hand.H11)):
    for _j, _c in enumerate(range(Card._2, Card._A)):
        fsm_hit[_h, _c] = _h + 2 + _j
    fsm_hit[_h, Card._A] = State.S13 + _i

for _j, _c in enumerate(range(Card._2, Card._A)):
    fsm_hit[Hand.H11, _c] = State.H13 + _j
fsm_hit[Hand.H11, Card._A] = State.H12

for _i, _h in enumerate(range(Hand.H12, Hand.H21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        fsm_hit[_h, _c] = _h + 2 + _j
    fsm_hit[_h, (Card._T - _i):Card._A] = State._BUST
    fsm_hit[_h, Card._A] = _h + 1

fsm_hit[Hand.H21, :] = State._BUST

for _j, _c in enumerate(range(Card._2, Card._A)):
    fsm_hit[Hand.T, _c] = State.H12 + _j
fsm_hit[Hand.T, Card._A] = State.BJ

for _j, _c in enumerate(range(Card._2, Card._T)):
    fsm_hit[Hand.A, _c] = State.S13 + _j
fsm_hit[Hand.A, Card._T] = State.BJ
fsm_hit[Hand.A, Card._A] = State.S12

for _i, _s in enumerate(range(Hand.S12, Hand.S21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        fsm_hit[_s, _c] = _s + 2 + _j
    for _j, _c in enumerate(range(Card._T - _i, Card._A)):
        fsm_hit[_s, _c] = State.H12 + _j
    fsm_hit[_s, Card._A] = _s + 1

for _c, _j in enumerate(range(Card._2, Card._A)):
    fsm_hit[Hand.S21, _c] = State.H13 + _j
fsm_hit[Hand.S21, Card._A] = State.H12

fsm_hit[Hand.BJ, :] = fsm_hit[Hand.S21, :]

# Finite-state machine for going from one hand to the next state after 'standing'.
fsm_stand = np.zeros(len(Hand), dtype=int)

fsm_stand[Hand.DEAL:Hand.H17] = State._16

for _i, _h in enumerate(range(Hand.H17, Hand.H21 + 1)):
    fsm_stand[_h] = State._17 + _i
fsm_stand[Hand.T:Hand.S17] = State._16

for _i, _s in enumerate(range(Hand.S17, Hand.S21 + 1)):
    fsm_stand[_s] = State._17 + _i
fsm_stand[Hand.BJ] = State._BJ

# Map a Hand to a Count
count = np.zeros(len(State), dtype=int)

count[:len(Hand)] = fsm_stand - len(Hand)

for _s in range(State._BUST, State._BJ + 1):
    count[_s] = _s - len(Hand)

# Deterministic policy for a dealer who stands on 17.
stand_on_17 = np.full(len(Hand), Action.h)

for _h in range(Hand.H17, Hand.H21 + 1):
    stand_on_17[_h] = Action.s

for _s in range(Hand.S17, Hand.BJ + 1):
    stand_on_17[_s] = Action.s

# Deterministic policy for a dealer who hits on soft 17.
hit_on_soft_17 = stand_on_17.copy()
hit_on_soft_17[Hand.S17] = Action.h

class InfiniteDeck:
    """
    An infinite deck of cards (i.e. drawing from a single suit with replacement).
    """
    def __init__(self, np_random):
        self.cards = np.array([ c for c in range(Card._2, Card._T) ] + [ Card._T ] * 4 + [ Card._A ])
        self.np_random = np_random

    def draw(self):
        """
        Draw a single card.
        """
        return self.np_random.choice(self.cards, replace=True)

    def deal(self):
        """
        Draw two player cards and one dealer card.
        """
        return self.draw(), self.draw(), self.draw()

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
        There are 34 * 10 = 340 discrete states:
            34 player counts (DEAL, H2-H21, T, A, S12-S21, BJ)
            10 dealer cards showing (2-9, T, A)

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

    def __init__(self, winning_blackjack=1., blackjack_beats_21=True, dealer_hits_on_soft_17=False, deck=InfiniteDeck):
        """
        Initialize the state of the environment.

        Args:
            winning_blackjack (float): how much a winning blackjack pays out. Defaults to 1.0.
            blackjack_beats_21 (bool): whether a blackjack (21 on the first two cards) beats a regular 21. Defaults to True.
            dealer_hits_on_soft (bool): whether the dealer stands or hits on soft 17. Defaults to False.
            deck (object): the class name of the deck. Defaults to 'InfiniteDeck'.

        Notes:
            The default arguments correspond to Sutton and Barto's 'Reinforcement Learning' (2018).
            blackjack_beats_21=False corresponds to the OpenAI Gym environment 'Blackjack-v0'.
            blackjack_beats_21=False with winning_blackjack=1.5 correponds to 'Blackjack-v0' initialized with natural=True.
            Casinos usually have winning_blackjack=1.5. Sometimes winning_blackjack=1.2 and/or dealer_hits_on_soft_17=True.
        """
        self.payout = np.zeros((len(Count), len(Count)))      # The player and dealer have equal scores.
        self.payout[Count._BUST, :          ]          = -1.  # The player busts regardless of whether the dealer busts.
        self.payout[Count._16:,  Count._BUST]          = +1.  # The dealer busts and the player doesn't.
        self.payout[np.tril_indices(len(Count), k=-1)] = +1.  # The player scores higher than the dealer.
        self.payout[np.triu_indices(len(Count), k=+1)] = -1.  # The dealer scores higher than the player.
        self.payout[Count._BJ,   :Count._BJ ]          = winning_blackjack
        if not blackjack_beats_21:
            self.payout[Count._BJ, Count._21]          =  0.  # A player's blackjack and a dealer's 21 are treated equally.
            self.payout[Count._21, Count._BJ]          =  0.  # A player's 21 and a dealer's blackjack are treated equally.
        self.dealer_policy = hit_on_soft_17 if dealer_hits_on_soft_17 else stand_on_17
        self.observation_space = spaces.Tuple((
            spaces.Discrete(len(Hand)), # only include transient Markov states because absorbing states don't need to be explored
            spaces.Discrete(len(Card))
        ))
        self.action_space = spaces.Discrete(len(Action))
        self.reward_range = (np.min(self.payout), np.max(self.payout))
        self.seed()
        self.deck = deck(self.np_random)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        p = state_labels[self.player]
        upcard_only = self.dealer in range(Card._2, Card._A + 1)
        if self.player != State._BUST and upcard_only:
            d = card_labels[self.dealer]
            return f'player: {p:>4}; dealer: {d:>4};'
        else:
            d = state_labels[fsm_hit[State.DEAL, self.dealer] if upcard_only else self.dealer]
            R = self.payout[count[self.player], count[self.dealer]]
            return f'player: {p:>4}; dealer: {d:>4}; reward: {R:>+4}'

    def _get_obs(self):
        return self.player, self.dealer

    def reset(self):
        """
        Reset the state of the environment by drawing two player cards and one dealer card.

        Returns:
            observation (object): the player's hand and the dealer's upcard.

        Notes:
            Cards are drawn from a single deck with replacement (i.e. from an infinite deck).
        """
        p1, p2, up = self.deck.deal()
        self.info = { 'player': [ card_labels[p1], card_labels[p2] ], 'dealer': [ card_labels[up] ] }
        self.player, self.dealer = fsm_hit[fsm_hit[Hand.DEAL, p1], p2], up
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
        if action:
            card = self.deck.draw()
            self.info['player'].append(card_labels[card])
            self.player = fsm_hit[self.player, card]
            done = self.player == State._BUST
        else:
            self.dealer = fsm_hit[Hand.DEAL, self.dealer]
            while True:
                card = self.deck.draw()
                self.info['dealer'].append(card_labels[card])
                self.dealer = fsm_hit[self.dealer, card]
                if self.dealer == State._BUST or not self.dealer_policy[self.dealer]:
                    break
            done = True
        reward = self.payout[count[self.player], count[self.dealer]] if done else 0.
        return self._get_obs(), reward, done, self.info
