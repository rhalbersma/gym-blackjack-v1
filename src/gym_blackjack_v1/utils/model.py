#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

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
    # Construct an absorbing Markov chain from the FSM and the card probabilities
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain
    P = markov_state_transitions(fsm, prob)
    Q = P[:-nT, :-nT]
    R = P[:-nT, -nT:]

    # Compute the absorbing probabilities
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Absorbing_probabilities
    I = np.identity(nM - nT)
    N = np.linalg.inv(I - Q)
    B = N @ R
    assert np.isclose(B.sum(axis=-1), 1).all()
    return B


def build(env):
    nS = nH * nC
    done = -1

    # Compute the initial state distribution.
    card_prob = env.deck.prob
    prob_m_a_m = markov_state_action_transitions(fsm.stand_hit, card_prob)
    hand_prob = np.linalg.matrix_power(prob_m_a_m[:, Action.HIT, :], 2)[Markov._DEAL, :nH]
    isd = (hand_prob.reshape(-1, 1) * card_prob.reshape(1, -1)).reshape(nS)
    assert np.isclose(isd.sum(), 1)

    dealer_one_hot = one_hot_encode(np.resize(env.dealer_policy, nM))
    dealer_fsm = fsm_policy(fsm.stand_hit, dealer_one_hot)
    dealer_terminal = card_transitions() @ absorbing_prob(dealer_fsm, card_prob)

    Reward = np.unique(env.payoff)
    nR = len(Reward)
    no_reward = np.where(Reward == 0)[0][0]

    prob_h_c_a_r_h_c = np.zeros((nH, nC, nA, nR, nH, nC))
    for c in Card:
        prob_h_c_a_r_h_c[:, c, Action.HIT, no_reward, :, c] = prob_m_a_m[:nH, Action.HIT, :nH]

    prob_h_c_a_r = np.zeros((nH, nC, nA, nR))
    for c in Card:
        for ri, r in enumerate(Reward):
            prob_h_c_a_r[:, c, :, ri] = prob_m_a_m[:nH, :, -nT:] @ (env.payoff == r) @ dealer_terminal[c].T

    # p(s', r|s, a): probability of transition to state s' with reward r, from state s and action a
    model = np.zeros((nS + 1, nA, nR, nS + 1))
    model[:done, Action.HIT, no_reward, :done] = prob_h_c_a_r_h_c[:, :, Action.HIT, no_reward, :, :].reshape((nS, nS))
    model[:done, :,          :,          done] = prob_h_c_a_r.reshape((nS, nA, nR))
    model[ done, :,          no_reward,  done] = 1
    assert np.isclose(model.sum(axis=(2, 3)), 1).all()

    # p(s'|s, a): probability of transition to state s', from state s taking action a
    transition = model.sum(axis=2)
    assert np.isclose(transition.sum(axis=2), 1).all()

    # r(s, a): expected immediate reward from state s after action a
    reward = model.sum(axis=3) @ Reward

    # OpenAI's Gym DiscreteEnv expects a dictionary of lists, where
    # P[s][a] == [(probability, nextstate, reward, done), ...]
    # In other words: P is a sparse representation of our dense tensor model[s, a, reward, nextstate]
    P = {
        s: {
            a: [
                (model[s, a, ri, next], next, r, next == done)
                for ri, r in enumerate(Reward)
                for next in range(done, nS)
                if model[s, a, ri, next] > 0
            ]
            for a in range(nA)
        }
        for s in range(nS)
    }

    return nS, nA, P, isd, transition, reward

