#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from ..enums import Action, Card, Hand, Markov, Terminal, card_labels, markov_labels, nA, nC, nH, nT
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
        There are 29 * 10 = 290 discrete observable states:
            29 player hands: H4-H21, S12-S21, BJ
            10 dealer cards: 2-9, T, A

    Actions:
        There are 2 actions for the player:
            0: stop the game ('stand' or 'stick') and play out the dealer's hand.
            1: request an additional card to be dealt ('hit').
        Note: additional actions allowed in casino play, such as taking insurance,
        doubling down, splitting or surrendering, are not supported.

    Rewards:
        There are up to 4 rewards for the player. In units relative to the bet size:
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

    def __init__(self, winning_blackjack_payoff=+1, blackjack_ties_with_21=False, dealer_hits_on_soft_17=False):
        """
        Initialize the state of the environment.

        Args:
            winning_blackjack_payoff (float): how much a winning blackjack pays out. Defaults to +1.
            blackjack_ties_with_21 (bool): whether a blackjack (21 on the first two cards) ties with a regular 21. Defaults to False.
            dealer_hits_on_soft_17 (bool): whether the dealer stands or hits on soft 17. Defaults to False.

        Notes:
            The default arguments correspond to Sutton and Barto's example in 'Reinforcement Learning' (2018).
            blackjack_ties_with_21=True corresponds to the OpenAI Gym environment gym.make('Blackjack-v0').
            blackjack_ties_with_21=True with winning_blackjack_payoff=1.5 correponds to gym.make('Blackjack-v0', natural=True).
            Casinos usually have winning_blackjack_payoff=+1.5 (sometimes +1.2), and/or dealer_hits_on_soft_17=True.
        """
        self.payoff = np.zeros((nT, nT))                    # The player and dealer have equal scores.
        self.payoff[Terminal._BUST, :             ] = -1    # The player busts regardless of whether the dealer busts.
        self.payoff[Terminal._16:,  Terminal._BUST] = +1    # The dealer busts and the player doesn't.
        self.payoff[np.tril_indices(nT, k=-1)     ] = +1    # The player scores higher than the dealer.
        self.payoff[np.triu_indices(nT, k=+1)     ] = -1    # The dealer scores higher than the player.
        self.payoff[Terminal._BJ,   :Terminal._BJ ] = winning_blackjack_payoff
        if blackjack_ties_with_21:
            self.payoff[Terminal._BJ, Terminal._21] =  0    # A player's blackjack and a dealer's 21 are treated equally.
            self.payoff[Terminal._21, Terminal._BJ] =  0    # A player's 21 and a dealer's blackjack are treated equally.
        self.reward_range = (np.min(self.payoff), np.max(self.payoff))
        self.dealer_policy = dealer.hits_on_soft_17 if dealer_hits_on_soft_17 else dealer.stands_on_17
        self.seed()
        self.deck = InfiniteDeck(self.np_random)
        self.nS, self.nA, self.P, self.isd, self.transition, self.reward = model.build(self)
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.observation_shape = (nH, nC)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        self.info = { 
            'player': [ card_labels[p1], card_labels[p2] ], 
            'dealer': [ card_labels[up] ] 
        }
        self.player, self.dealer = fsm.hit[fsm.hit[Markov._DEAL, p1], p2], up
        return self._get_obs()

    def explore(self, start):
        """
        Explore a specific starting state of the environment.

        Args:
            start (tuple of ints): the player's hand and the dealer's upcard.

        Returns:
            observation (object): the player's hand and the dealer's upcard.

        Notes:
            This is an extension of the OpenAI Gym interface.
            Monte Carlo Exploring Starts should use this method instead of reset().
        """
        self.player, self.dealer = start
        self.info = { 
            'player': [], 
            'dealer': [] 
        }
        return self._get_obs()

    def step(self, action):
        if action:
            card = self.deck.draw()
            self.info['player'].append(card_labels[card])
            self.player = fsm.hit[self.player, card]
            done = self.player == Markov._BUST
        else:
            self.dealer = fsm.hit[Markov._DEAL, self.dealer]
            while True:
                card = self.deck.draw()
                self.info['dealer'].append(card_labels[card])
                self.dealer = fsm.hit[self.dealer, card]
                if self.dealer == Markov._BUST or not self.dealer_policy[self.dealer]:
                    break
            done = True
        reward = self.payoff[fsm.count[self.player], fsm.count[self.dealer]] if done else 0
        return self._get_obs(), reward, done, self.info

    def render(self, mode='human'):
        p = markov_labels[self.player]
        upcard_only = self.dealer in range(Card._2, Card._A + 1)
        if self.player != Markov._BUST and upcard_only:
            d = card_labels[self.dealer]
            return f'player: {p:>4}; dealer: {d:>4};'
        else:
            d = markov_labels[fsm.hit[Markov._DEAL, self.dealer] if upcard_only else self.dealer]
            R = self.payoff[fsm.count[self.player], fsm.count[self.dealer]]
            return f'player: {p:>4}; dealer: {d:>4}; reward: {R:>+4}'

