#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

from gym_blackjack_v1.enums import Action, Card, Count, Hand, State
from gym_blackjack_v1.utils import fsm

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


def state_transitions(fsm, prob):
    p = np.zeros((len(State), len(State)))
    for s0, successors in enumerate(fsm):
        for card, s1 in enumerate(successors):
            p[s0, s1] += prob[card]
    assert np.isclose(p.sum(axis=-1), 1).all()
    return p


def state_action_transitions(fsm_array, prob):
    p = np.zeros((len(State), len(Action), len(State)))
    for a in Action:
        p[:, a, :] = state_transitions(fsm_array[a], prob)
    assert np.isclose(p.sum(axis=-1), 1).all()
    return p


def upcard_transitions():
    p = np.zeros((len(Card), len(State) - len(Count)))
    for uc in Card:
        h = fsm.hit[State._DEAL, uc]
        p[uc, h] = 1
    assert np.isclose(p.sum(axis=-1), 1).all()
    return p


def upcard_counts(fsm, prob):
    # Construct an absorbing Markov chain from the FSM and the card probabilities
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain
    P_s_s = state_transitions(fsm, prob)
    Q_h_h = P_s_s[:-len(Count), :-len(Count)]
    R_h_c = P_s_s[:-len(Count), -len(Count):]
    I_h_h = np.identity(len(State) - len(Count))

    # Compute the absorbing probabilities
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Absorbing_probabilities 
    N_h_h = np.linalg.inv(I_h_h - Q_h_h)
    B_h_c = N_h_h @ R_h_c

    prob_uc_h = upcard_transitions()
    p = prob_uc_h @ B_h_c
    assert np.isclose(p.sum(axis=-1), 1).all()
    return p


def one_hot_policy(policy):
    one_hot = np.zeros((policy.size, policy.max() + 1), dtype=int)
    one_hot[np.arange(policy.size), policy] = 1
    return one_hot


def build(payout, dealer_policy, prob):
    fsm_array = np.array([a for a in np.broadcast_arrays(fsm.stand.reshape(-1, 1), fsm.hit)])
    prob_s_a_s = state_action_transitions(fsm_array, prob)

    dealer_policy_one_hot = one_hot_policy(np.resize(dealer_policy, len(State)))
    dealer_fsm = (fsm_array * np.expand_dims(dealer_policy_one_hot, axis=0).T).sum(axis=0)
    prob_c_uc = upcard_counts(dealer_fsm, prob).T

    Reward = np.unique(payout)
    no_reward = np.where(Reward == 0)[0][0]

    prob_h_c_a_r_h_c = np.zeros((len(Hand), len(Card), len(Action), len(Reward), len(Hand), len(Card)))
    for _uc in Card:
        prob_h_c_a_r_h_c[:, _uc, Action.HIT, no_reward, :, _uc] = prob_s_a_s[:len(Hand), Action.HIT, :len(Hand)]

    prob_h_c_a_r = np.zeros((len(Hand), len(Card), len(Action), len(Reward)))
    for _uc in Card:
        for _i, _r in enumerate(Reward):
            prob_h_c_a_r[:, _uc, :, _i] = prob_s_a_s[:len(Hand), :, -len(Count):] @ (payout == _r) @ prob_c_uc[:, _uc]

    # p(s', r|s, a): probability of transition to state s' with reward r, from state s and action a
    model = np.zeros((len(Hand) * len(Card) + 1, len(Action), len(Reward), len(Hand) * len(Card) + 1))    
    model[:-1, Action.HIT, no_reward, :-1] = prob_h_c_a_r_h_c[:, :, Action.HIT, no_reward, :, :].reshape((len(Hand) * len(Card), len(Hand) * len(Card)))
    model[:-1, :,          :,          -1] = prob_h_c_a_r[:, :, :, :].reshape((len(Hand) * len(Card), len(Action), len(Reward)))
    model[ -1, :,          no_reward,  -1] = 1
    assert np.isclose(model.sum(axis=(2, 3)), 1).all()

    # p(s'|s, a): probability of transition to state s', from state s taking action a
    transition = model.sum(axis=2)
    assert np.isclose(transition.sum(axis=2), 1).all()

    # r(s, a): expected immediate reward from state s after action a
    reward = model.sum(axis=3) @ Reward

    return model, transition, reward

