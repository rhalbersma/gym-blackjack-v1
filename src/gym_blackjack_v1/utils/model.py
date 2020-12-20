#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

from ..enums import Action, Card, Dealer, Hand, Player, Terminal
from . import fsm

################################################################################
# Blackjack with an InfiniteDeck is a Markov Decision Problem (MDP).
# We can construct a full model for this environment through the definition of
# prob_s_a_r_s' = a rank-4 tensor containing the probabilities of starting with
# state s, applying action a, receiving reward r and transitioning to state s'.
################################################################################


def player_card_transitions(fsm, prob):
    prob_p_p = np.zeros((len(Player), len(Player)))
    for p0, successors in enumerate(fsm):
        for card, p1 in enumerate(successors):
            prob_p_p[p0, p1] += prob[card]
    assert np.isclose(prob_p_p.sum(axis=-1), 1).all()
    return prob_p_p


def player_action_transitions(fsm, prob):
    prob_p_a_p = np.zeros((len(Player), len(Action), len(Player)))
    for a in Action:
        prob_p_a_p[:, a, :] = player_card_transitions(fsm[a], prob)
    assert np.isclose(prob_p_a_p.sum(axis=-1), 1).all()
    return prob_p_a_p


def upcard_hand_transitions(fsm):
    prob_uc_h = np.zeros((len(Card), len(Hand)))
    for card in range(len(Card)):
        player = fsm[Player.NONE, card]
        prob_uc_h[card, player] = 1
    assert np.isclose(prob_uc_h.sum(axis=-1), 1).all()
    return prob_uc_h


def one_hot_policy(policy):
    one_hot = np.zeros((policy.size, policy.max() + 1), dtype=int)
    one_hot[np.arange(policy.size), policy] = 1
    return one_hot


def build(payout, dealer_policy, prob):
    fsm_array = np.array([ 
        a for a in np.broadcast_arrays(
            fsm.stand.reshape(-1, 1), 
            fsm.hit
        ) 
    ])
    prob_p_a_p = player_action_transitions(fsm_array, prob)
    dealer_policy_one_hot = one_hot_policy(np.resize(dealer_policy, len(Player)))
    dealer_fsm = (fsm_array * np.expand_dims(dealer_policy_one_hot, axis=0).T).sum(axis=0)
    id_t = np.identity(len(Player) - len(Terminal))
    dealer_N = np.linalg.inv(id_t - player_card_transitions(dealer_fsm, prob)[:-len(Terminal), :-len(Terminal)])
    prob_uc_h = upcard_hand_transitions(fsm_array[Action.h])
    prob_uc_c = prob_uc_h @ dealer_N[:len(Hand), len(Hand):]
    reward_values = np.unique(payout)
    reward_outcomes = np.array([
        payout == r
        for r in reward_values
    ], dtype=int)
    prob_c_uc_r = (reward_outcomes @ prob_uc_c.T).transpose(1, 2, 0)

    model = np.zeros((len(Player), len(Dealer), len(Action), len(reward_values), len(Player), len(Dealer)))
    for _c in range(len(Card)):
        model[:len(Hand), _c, :, 1, :-len(Terminal), _c] = prob_p_a_p[:len(Hand), :, :-len(Terminal)]
    for _a in range(len(Action)):
        model[len(Hand):-len(Terminal), :-len(Terminal), _a, :, Player._END, Dealer._END] = (prob_p_a_p[len(Hand):-len(Terminal), _a, Player._END] * prob_c_uc_r.T).T
    model[Player._END, Dealer._END, :, 1, Player._END, Dealer._END] = prob_p_a_p[Player._END, :, Player._END]

    assert np.isclose(model[..., -len(Terminal):, :-len(Terminal)], 0).all()
    assert np.isclose(model[..., :-len(Terminal), -len(Terminal):], 0).all()

    model[Player._END, :-len(Terminal), :, 1, Player._END, Dealer._END] = 1
    model[:-len(Terminal), Dealer._END, :, 1, Player._END, Dealer._END] = 1

    # p(s', r|s, a): probability of transition to state s' with reward r, from state s and action a
    model = model.reshape((len(Player) * len(Dealer), len(Action), len(reward_values), len(Player) * len(Dealer)))
    assert np.isclose(model.sum(axis=(2, 3)), 1).all()

    # p(s'|s, a): probability of transition to state s', from state s taking action a
    transition = model.sum(axis=2)
    assert np.isclose(transition.sum(axis=2), 1).all()

    # r(s, a): expected immediate reward from state s after action a
    reward = model.sum(axis=3) @ reward_values

    return model, transition, reward

