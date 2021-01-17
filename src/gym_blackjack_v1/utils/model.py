#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from itertools import product

import numpy as np

from ..enums import Action, Card, Hand, Markov, Terminal, nA, nC, nH, nM, nT
from ..utils import fsm

################################################################################
# Although we have complete knowledge of the environment in the blackjack task,
# it would not be easy to apply DP methods to compute the value function.
# DP methods require the distribution of next events — in particular, they
# require the environments dynamics as given by the four-argument function p —
# and it is not easy to determine this for blackjack. For example, suppose the
# player’s sum is 14 and he chooses to stick. What is his probability of
# terminating with a reward of +1 as a function of the dealer’s showing card?
# All of the probabilities must be computed before DP can be applied, and such
# computations are often complex and error-prone.
#
# Sutton & Barto, Reinforcement Learning, p. 94.
################################################################################


def inifinite_deck():
    cards = np.array([ c for c in range(Card._2, Card._T) ] + 4 * [ Card._T ] + [ Card._A ])
    _, counts = np.unique(cards, return_counts=True)
    p = counts / counts.sum()
    assert np.isclose(p.sum(), 1)
    return p


def markov_state_transitions(fsm, prob):
    p = np.zeros((nM, nM))
    for m0, successors in enumerate(fsm):
        for card, m1 in enumerate(successors):
            p[m0, m1] += prob[card]
    assert np.isclose(p.sum(axis=-1), 1).all()
    return p


def markov_state_action_transitions(fsm_array, prob):
    p = np.zeros((nM, nA, nM))
    for a in Action:
        p[:, a, :] = markov_state_transitions(fsm_array[a], prob)
    assert np.isclose(p.sum(axis=-1), 1).all()
    return p


def card_transitions():
    p = np.zeros((nC, nM - nT))
    for c in Card:
        h = fsm.hit[Markov._DEAL, c]
        p[c, h] = 1
    assert np.isclose(p.sum(axis=-1), 1).all()
    return p


def one_hot_encode(policy):
    one_hot = np.zeros((policy.size, policy.max() + 1), dtype=int)
    one_hot[np.arange(policy.size), policy] = 1
    return one_hot


def fsm_policy(fsm_array, one_hot_policy):
    return (fsm_array * np.expand_dims(one_hot_policy, axis=0).T).sum(axis=0)


def absorbing_prob(fsm, prob):
    # Construct an absorbing Markov chain from the FSM and the card probabilities.
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain
    P = markov_state_transitions(fsm, prob)
    Q = P[:-nT, :-nT]
    R = P[:-nT, -nT:]

    # Compute the absorbing probabilities.
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Absorbing_probabilities
    I = np.identity(nM - nT)
    N = np.linalg.inv(I - Q)
    B = N @ R
    assert np.isclose(B.sum(axis=-1), 1).all()
    return B


def build(env):
    nS  = nH * nC   # The number of non-terminal states: |S|.

    # Compute the initial state distribution.
    card_prob = inifinite_deck()
    prob_m_a_m = markov_state_action_transitions(fsm.stand_hit, card_prob)
    hand_prob = np.linalg.matrix_power(prob_m_a_m[:, Action.HIT, :], 2)[Markov._DEAL, :nH]
    reset_pdf = (hand_prob.reshape(-1, 1) * card_prob.reshape(1, -1)).reshape(nS)
    assert np.isclose(reset_pdf.sum(), 1)
    reset_cdf = reset_pdf.cumsum()

    # Compute the terminal state distributions.
    # For the dealer, it's conditional on his upcard.
    # For the player, it's conditional on his hand and action.
    dealer_one_hot = one_hot_encode(np.resize(env.dealer_policy, nM))
    dealer_fsm = fsm_policy(fsm.stand_hit, dealer_one_hot)
    dealer_terminal = card_transitions() @ absorbing_prob(dealer_fsm, card_prob)
    player_terminal = prob_m_a_m[:nH, :, -nT:]

    unique_rewards = np.unique(env.payoff)
    nR = len(unique_rewards)
    no_reward = np.where(unique_rewards == 0)[0][0]

    prob_h_c_a_h_c_r = np.zeros((nH, nC, nA, nH, nC, nR))
    for c in range(nC):
        prob_h_c_a_h_c_r[:, c, Action.HIT, :, c, no_reward] = prob_m_a_m[:nH, Action.HIT, :nH]
    prob_sHs0 = prob_h_c_a_h_c_r[:, :, Action.HIT, :, :, no_reward].reshape((nS, nS))

    prob_h_c_a_r = np.zeros((nH, nC, nA, nR))
    for c, r in product(range(nC), range(nR)):
        prob_h_c_a_r[:, c, :, r] = player_terminal @ (env.payoff == unique_rewards[r]) @ dealer_terminal[c].T
    prob_saTr = prob_h_c_a_r.reshape((nS, nA, nR))

    # OpenAI's Gym DiscreteEnv expects a dictionary of lists, where
    # P[s][a] == [(prob, next, reward, done), ...]
    # In other words: P is a sparse representation of Sutton & Barto's dense 4-tensor p(s', r|s, a).
    P = { 
        s: {
            Action.HIT: [
                (prob_sHs0[s, next], next, unique_rewards[no_reward], False)
                for next in range(nS)
                if prob_sHs0[s, next] > 0
            ] + [
                (prob_saTr[s, Action.HIT, r], -1, unique_rewards[r], True)
                for r in range(nR)
                if prob_saTr[s, Action.HIT, r] > 0
            ],
            Action.STAND: [
                (prob_saTr[s, Action.STAND, r], -1, unique_rewards[r], True)
                for r in range(nR)
                if prob_saTr[s, Action.STAND, r] > 0
            ]
        }
        for s in range(nS)
    }

    # Equation (3.4) in Sutton & Barto (p.49):
    # p(s'|s, a) = probability of transition to state s', from state s taking action a.
    transition = np.zeros((nS, nA, nS))

    # Equation (3.5) in Sutton & Barto (p.49):
    # r(s, a) = expected immediate reward from state s after action a.
    reward = np.zeros((nS, nA))

    # Initialize the transition and reward tensors.
    for s in P.keys():
        for a in P[s].keys():
            for prob, next, r, done in P[s][a]:
                # Exclude transitions to the terminal state.
                if not done:
                    transition[s, a, next] += prob
                reward[s, a] += prob * r

    step_cdf = {
        s: {
            a: np.array([
                t[0]
                for t in P[s][a]
            ]).cumsum()
            for a in P[s].keys()
        }
        for s in P.keys()
    }

    return nS, nA, P, reset_pdf, reset_cdf, step_cdf, transition, reward

