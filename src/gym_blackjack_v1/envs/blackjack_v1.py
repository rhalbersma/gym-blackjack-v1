#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from aenum import IntEnum, extend_enum
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd

################################################################################
# Enumerating the state and action spaces
################################################################################

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

class Card(IntEnum):
    _2 = 0
    _3 = 1
    _4 = 2
    _5 = 3
    _6 = 4
    _7 = 5
    _8 = 6
    _9 = 7
    _T = 8 # 10, J, Q, K are all denoted as T
    _A = 9

card_labels = [ c.name[1:] for c in Card ]

class Terminal(IntEnum):
    _END = 0

terminal_labels = [ t.name[1:] for t in Terminal ]

class Player(IntEnum):
    pass

for key, value in Hand.__members__.items():
    extend_enum(Player, key, value)
for key, value in Count.__members__.items():
    extend_enum(Player, key, value + len(Hand))
for key, value in Terminal.__members__.items():
    extend_enum(Player, key, value + len(Hand) + len(Count))

player_labels = hand_labels + count_labels + terminal_labels

class Dealer(IntEnum):
    pass

for key, value in Card.__members__.items():
    extend_enum(Dealer, key, value)
for key, value in Terminal.__members__.items():
    extend_enum(Dealer, key, value + len(Card))

dealer_labels = card_labels + terminal_labels

class Action(IntEnum):
    s = 0
    h = 1

action_labels = [ a.name[0] for a in Action ]

################################################################################
# We factor all transition logic into a finite-state machine (FSM) lookup table.
# Note that this is model-free RL because there are no explicit probabilities.
#
# The FSM approach has two main benefits:
# 1) it speeds up the model-free version of the BlackjackEnv class;
# 2) it is straightforward to extend to a model-based version.
################################################################################

# Going from a state to the next state after 'hitting' another card.
fsm_hit = np.zeros((len(Player), len(Card)), dtype=int)

for _j, _c in enumerate(range(Card._2, Card._T)):
    fsm_hit[Player.NONE, _c] = Player.H2 + _j
fsm_hit[Player.NONE, Card._T] = Player.T
fsm_hit[Player.NONE, Card._A] = Player.A

for _i, _h in enumerate(range(Player.H2, Player.H11)):
    for _j, _c in enumerate(range(Card._2, Card._A)):
        fsm_hit[_h, _c] = _h + 2 + _j
    fsm_hit[_h, Card._A] = Player.S13 + _i

for _j, _c in enumerate(range(Card._2, Card._A)):
    fsm_hit[Player.H11, _c] = Player.H13 + _j
fsm_hit[Player.H11, Card._A] = Player.H12

for _i, _h in enumerate(range(Player.H12, Player.H21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        fsm_hit[_h, _c] = _h + 2 + _j
    fsm_hit[_h, (Card._T - _i):Card._A] = Player._BUST
    fsm_hit[_h, Card._A] = _h + 1

fsm_hit[Player.H21, :] = Player._BUST

for _j, _c in enumerate(range(Card._2, Card._A)):
    fsm_hit[Player.T, _c] = Player.H12 + _j
fsm_hit[Player.T, Card._A] = Player.BJ

for _j, _c in enumerate(range(Card._2, Card._T)):
    fsm_hit[Player.A, _c] = Player.S13 + _j
fsm_hit[Player.A, Card._T] = Player.BJ
fsm_hit[Player.A, Card._A] = Player.S12

for _i, _s in enumerate(range(Player.S12, Player.S21)):
    for _j, _c in enumerate(range(Card._2, Card._T - _i)):
        fsm_hit[_s, _c] = _s + 2 + _j
    for _j, _c in enumerate(range(Card._T - _i, Card._A)):
        fsm_hit[_s, _c] = Player.H12 + _j
    fsm_hit[_s, Card._A] = _s + 1

for _c, _j in enumerate(range(Card._2, Card._A)):
    fsm_hit[Player.S21, _c] = Player.H13 + _j
fsm_hit[Player.S21, Card._A] = Player.H12

fsm_hit[Player.BJ, :] = fsm_hit[Player.S21, :]

fsm_hit[Player._BUST:, :] = Player._END

# Going from one state to the next state after 'standing'.
fsm_stand = np.zeros(len(Player), dtype=int)

fsm_stand[:Player.H17] = Player._16

for _i, _h in enumerate(range(Player.H17, Player.H21 + 1)):
    fsm_stand[_h] = Player._17 + _i
fsm_stand[Player.T:Player.S17] = Player._16

for _i, _s in enumerate(range(Player.S17, Player.S21 + 1)):
    fsm_stand[_s] = Player._17 + _i
fsm_stand[Player.BJ] = Player._BJ

fsm_stand[Player._BUST:] = Player._END

# Going from a Hand to a Count.
count = np.zeros(len(Player), dtype=int)

count[:len(Hand)] = fsm_stand[:len(Hand)] - len(Hand)

for _s in range(Player._BUST, Player._BJ + 1):
    count[_s] = _s - len(Hand)

count[Player._END] = count[Player._BUST]

################################################################################
# Dealer policies
################################################################################

# Deterministic policy for a dealer who stands on 17.
stand_on_17 = np.full(len(Hand), Action.h)

for _h in range(Hand.H17, Hand.H21 + 1):
    stand_on_17[_h] = Action.s

for _s in range(Hand.S17, Hand.BJ + 1):
    stand_on_17[_s] = Action.s

# Deterministic policy for a dealer who hits on soft 17.
hit_on_soft_17 = stand_on_17.copy()
hit_on_soft_17[Hand.S17] = Action.h

################################################################################
# The deck of cards
################################################################################

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

################################################################################
# Blackjack with an InfiniteDeck is a Markov Decision Problem (MDP).
# We can construct a full model for this environment through the definition of
# prob_s_a_r_s' = a rank-4 tensor containing the probabilities of starting with
# state s, applying action a, receiving reward r and transitioning to state s'.
################################################################################

# Card probabilities for an InfiniteDeck
prob = np.ones((len(Card))) / 13.
prob[Card._T] *= 4.  # 10, J, Q, K all count as T
assert np.isclose(prob.sum(), 1.)

# Transition probabilities from a finite-state machine

def player_card_transitions(fsm):
    prob_p_p = np.zeros((len(Player), len(Player)))
    for p0, successors in enumerate(fsm):
        for card, p1 in enumerate(successors):
            prob_p_p[p0, p1] += prob[card]
    assert np.isclose(prob_p_p.sum(axis=-1), 1.).all()
    return prob_p_p

def player_action_transitions(fsm):
    prob_p_a_p = np.zeros((len(Player), len(Action), len(Player)))
    for a in Action:
        prob_p_a_p[:, a, :] = player_card_transitions(fsm[a])
    assert np.isclose(prob_p_a_p.sum(axis=-1), 1.).all()
    return prob_p_a_p

def upcard_hand_transitions(fsm):
    prob_uc_h = np.zeros((len(Card), len(Hand)))
    for card in range(len(Card)):
        player = fsm[Player.NONE, card]
        prob_uc_h[card, player] = 1.
    assert np.isclose(prob_uc_h.sum(axis=-1), 1.).all()
    return prob_uc_h

fsm = np.array([ a for a in np.broadcast_arrays(fsm_stand.reshape(-1,1), fsm_hit) ])
prob_p_a_p = player_action_transitions(fsm)
prob_uc_h = upcard_hand_transitions(fsm[Action.h])

def one_hot_policy(policy):
    one_hot = np.zeros((policy.size, policy.max() + 1), dtype=int)
    one_hot[np.arange(policy.size), policy] = 1
    return one_hot

id_t = np.identity(len(Player) - len(Terminal))

################################################################################
# Environment
################################################################################

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
            34 player counts (NONE, H2-H21, T, A, S12-S21, BJ)
            10 dealer cards showing (2-9, T, A)

    Actions:
        There are 2 actions for the player:
            0: stop the game ('stand' or 'stick') and play out the dealer's hand.
            1: request an additional card to be dealt ('hit').
        Note: additional actions allowed in casino play, such as taking insurance,
        doubling down, splitting or surrendering, are not supported.

    Rewards:
        There are up to 4 rewards for the player:
            -1. : the player busted, regardless of the dealer,
                or the player's total is lower than the dealer's,
                or the dealer has a blackjack and the player doesn't;
             0. : the player's total is equal to the dealer's;
            +1. : the player's total is higher than the dealer's,
                or the dealer busted and the player didn't;
            +1.5: the player has a blackjack and the dealer doesn't.
                (in Sutton and Barto, this pays out +1. to the player).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, winning_blackjack=+1., blackjack_beats_21=True, dealer_hits_on_soft_17=False, deck=InfiniteDeck, model=False):
        """
        Initialize the state of the environment.

        Args:
            winning_blackjack (float): how much a winning blackjack pays out. Defaults to +1.
            blackjack_beats_21 (bool): whether a blackjack (21 on the first two cards) beats a regular 21. Defaults to True.
            dealer_hits_on_soft (bool): whether the dealer stands or hits on soft 17. Defaults to False.
            deck (object): the class name of the deck. Defaults to 'InfiniteDeck'.
            model (bool): whether reward and transition probability tensors should be computed. Defaults to False.

        Notes:
            The default arguments correspond to Sutton and Barto's example in 'Reinforcement Learning' (2018).
            blackjack_beats_21=False corresponds to the OpenAI Gym environment 'Blackjack-v0'.
            blackjack_beats_21=False with winning_blackjack=1.5 correponds to 'Blackjack-v0' initialized with natural=True.
            Casinos usually have winning_blackjack=1.5, and sometimes winning_blackjack=1.2 and/or dealer_hits_on_soft_17=True.
        """
        self.payout = np.zeros((len(Count), len(Count)))        # The player and dealer have equal scores.
        self.payout[Count._BUST, :          ]          = -1.    # The player busts regardless of whether the dealer busts.
        self.payout[Count._16:,  Count._BUST]          = +1.    # The dealer busts and the player doesn't.
        self.payout[np.tril_indices(len(Count), k=-1)] = +1.    # The player scores higher than the dealer.
        self.payout[np.triu_indices(len(Count), k=+1)] = -1.    # The dealer scores higher than the player.
        self.payout[Count._BJ,   :Count._BJ ]          = winning_blackjack
        if not blackjack_beats_21:
            self.payout[Count._BJ, Count._21]          =  0.    # A player's blackjack and a dealer's 21 are treated equally.
            self.payout[Count._21, Count._BJ]          =  0.    # A player's 21 and a dealer's blackjack are treated equally.
        self.dealer_policy = hit_on_soft_17 if dealer_hits_on_soft_17 else stand_on_17
        self.observation_space = spaces.Tuple((
            spaces.Discrete(len(Hand)),                         # Only include transient Markov states because
            spaces.Discrete(len(Card))                          # absorbing states don't need to be explored.
        ))
        self.action_space = spaces.Discrete(len(Action))
        self.reward_range = (np.min(self.payout), np.max(self.payout))
        self.seed()
        self.deck = deck(self.np_random)

        if isinstance(self.deck, InfiniteDeck) and model:
            self.state_space = spaces.Tuple((
                spaces.Discrete(len(Player)),                   # Include all transient and absorbing Markov states.
                spaces.Discrete(len(Dealer))
            ))
            dealer_policy_one_hot = one_hot_policy(np.resize(self.dealer_policy, len(Player)))
            dealer_fsm = (fsm * np.expand_dims(dealer_policy_one_hot, axis=0).T).sum(axis=0)
            dealer_N = np.linalg.inv(id_t - player_card_transitions(dealer_fsm)[:-len(Terminal), :-len(Terminal)])
            prob_uc_c = prob_uc_h @ dealer_N[:len(Hand), len(Hand):]
            self.reward_values = np.unique(self.payout)
            reward_outcomes = np.array([
                self.payout == r
                for r in self.reward_values
            ], dtype=int)
            prob_c_uc_r = (reward_outcomes @ prob_uc_c.T).transpose(1, 2, 0)

            self.model = np.zeros((len(Player), len(Dealer), len(Action), len(self.reward_values), len(Player), len(Dealer)))
            for _c in range(len(Card)):
                self.model[:len(Hand), _c, :, 1, :-len(Terminal), _c] = prob_p_a_p[:len(Hand), :, :-len(Terminal)]
            for _a in range(len(Action)):
                self.model[len(Hand):-len(Terminal), :-len(Terminal), _a, :, Player._END, Dealer._END] = (prob_p_a_p[len(Hand):-len(Terminal), _a, Player._END] * prob_c_uc_r.T).T
            self.model[Player._END, Dealer._END, :, 1, Player._END, Dealer._END] = prob_p_a_p[Player._END, :, Player._END]

            assert np.isclose(self.model[..., -len(Terminal):, :-len(Terminal)], 0.).all()
            assert np.isclose(self.model[..., :-len(Terminal), -len(Terminal):], 0.).all()

            self.model[Player._END, :-len(Terminal), :, 1, Player._END, Dealer._END] = 1.
            self.model[:-len(Terminal), Dealer._END, :, 1, Player._END, Dealer._END] = 1.

            self.model = self.model.reshape((len(Player) * len(Dealer), len(Action), len(self.reward_values), len(Player) * len(Dealer)))
            assert np.isclose(self.model.sum(axis=(2, 3)), 1.).all()

            self.transition = self.model.sum(axis=2)
            assert np.isclose(self.transition.sum(axis=2), 1.).all()

            self.reward = self.model.sum(axis=3) @ self.reward_values

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        p = player_labels[self.player]
        upcard_only = self.dealer in range(Card._2, Card._A + 1)
        if self.player != Player._BUST and upcard_only:
            d = dealer_labels[self.dealer]
            return f'player: {p:>4}; dealer: {d:>4};'
        else:
            d = player_labels[fsm_hit[Player.NONE, self.dealer] if upcard_only else self.dealer]
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
        self.player, self.dealer = fsm_hit[fsm_hit[Player.NONE, p1], p2], up
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
            done = self.player == Player._BUST
        else:
            self.dealer = fsm_hit[Player.NONE, self.dealer]
            while True:
                card = self.deck.draw()
                self.info['dealer'].append(card_labels[card])
                self.dealer = fsm_hit[self.dealer, card]
                if self.dealer == Player._BUST or not self.dealer_policy[self.dealer]:
                    break
            done = True
        reward = self.payout[count[self.player], count[self.dealer]] if done else 0.
        return self._get_obs(), reward, done, self.info
