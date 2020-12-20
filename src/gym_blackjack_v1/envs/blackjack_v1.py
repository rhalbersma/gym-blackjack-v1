#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from ..enums import Action, Card, Count, Dealer, Hand, Player, Terminal, card_labels, dealer_labels, player_labels
from ..utils import dealer, fsm, InfiniteDeck, model


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
            -1  : the player busted, regardless of the dealer,
                or the player's total is lower than the dealer's,
                or the dealer has a blackjack and the player doesn't;
             0  : the player's total is equal to the dealer's;
            +1  : the player's total is higher than the dealer's,
                or the dealer busted and the player didn't;
            +1.5: the player has a blackjack and the dealer doesn't.
                (in Sutton and Barto, this pays out +1 to the player).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, winning_blackjack=+1, blackjack_beats_21=True, dealer_hits_on_soft_17=False, deck=InfiniteDeck, model_based=False):
        """
        Initialize the state of the environment.

        Args:
            winning_blackjack (float): how much a winning blackjack pays out. Defaults to +1.
            blackjack_beats_21 (bool): whether a blackjack (21 on the first two cards) beats a regular 21. Defaults to True.
            dealer_hits_on_soft (bool): whether the dealer stands or hits on soft 17. Defaults to False.
            deck (object): the class name of the deck. Defaults to 'InfiniteDeck'.
            model_based (bool): whether reward and transition probability tensors should be computed. Defaults to False.

        Notes:
            The default arguments correspond to Sutton and Barto's example in 'Reinforcement Learning' (2018).
            blackjack_beats_21=False corresponds to the OpenAI Gym environment 'Blackjack-v0'.
            blackjack_beats_21=False with winning_blackjack=1.5 correponds to 'Blackjack-v0' initialized with natural=True.
            Casinos usually have winning_blackjack=1.5, and sometimes winning_blackjack=1.2 and/or dealer_hits_on_soft_17=True.
        """
        self.payout = np.zeros((len(Count), len(Count)))        # The player and dealer have equal scores.
        self.payout[Count._BUST, :          ]          = -1     # The player busts regardless of whether the dealer busts.
        self.payout[Count._16:,  Count._BUST]          = +1     # The dealer busts and the player doesn't.
        self.payout[np.tril_indices(len(Count), k=-1)] = +1     # The player scores higher than the dealer.
        self.payout[np.triu_indices(len(Count), k=+1)] = -1     # The dealer scores higher than the player.
        self.payout[Count._BJ,   :Count._BJ ]          = winning_blackjack
        if not blackjack_beats_21:
            self.payout[Count._BJ, Count._21]          =  0     # A player's blackjack and a dealer's 21 are treated equally.
            self.payout[Count._21, Count._BJ]          =  0     # A player's 21 and a dealer's blackjack are treated equally.
        self.dealer_policy = dealer.hits_on_soft_17 if dealer_hits_on_soft_17 else dealer.stands_on_17
        self.observation_space = spaces.Tuple((
            spaces.Discrete(len(Hand)),                         # Only include transient Markov states because
            spaces.Discrete(len(Card))                          # absorbing states don't need to be explored.
        ))
        self.action_space = spaces.Discrete(len(Action))
        self.reward_range = (np.min(self.payout), np.max(self.payout))
        self.seed()
        self.deck = deck(self.np_random)
        if model_based:
            self.build_model()
        self.reset()

    def build_model(self):
        if not isinstance(self.deck, InfiniteDeck):
            raise TypeError(f'To build an env\'s model, its deck attribute needs to be of class type "InfiniteDeck" rather than "{type(self.deck).__name__}".')
        self.state_space = spaces.Tuple((
            spaces.Discrete(len(Player)),   # Include all transient and absorbing Markov states because
            spaces.Discrete(len(Dealer))    # absorbing states will ensure termination of dynamic programming methods.
        ))
        self.model, self.transition, self.reward = model.build(self.payout, self.dealer_policy, self.deck.prob)

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
            d = player_labels[fsm.hit[Player.NONE, self.dealer] if upcard_only else self.dealer]
            R = self.payout[fsm.count[self.player], fsm.count[self.dealer]]
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
        self.player, self.dealer = fsm.hit[fsm.hit[Player.NONE, p1], p2], up
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
            self.player = fsm.hit[self.player, card]
            done = self.player == Player._BUST
        else:
            self.dealer = fsm.hit[Player.NONE, self.dealer]
            while True:
                card = self.deck.draw()
                self.info['dealer'].append(card_labels[card])
                self.dealer = fsm.hit[self.dealer, card]
                if self.dealer == Player._BUST or not self.dealer_policy[self.dealer]:
                    break
            done = True
        reward = self.payout[fsm.count[self.player], fsm.count[self.dealer]] if done else 0
        return self._get_obs(), reward, done, self.info

