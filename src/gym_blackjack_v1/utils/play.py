#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from collections import defaultdict

import gym
import statsmodels.stats.weightstats as ssw

from ..agents import BasicStrategyAgent
from ..enums import Action, action_labels


def play(env, episodes=100, hint=False):
    """
    Play an interactive session of blackjack.

    Args:
        env (object): an OpenAI Gym blackjack environment. Defaults to 'Blackjack-v1'.
        episodes (int): the number of episodes to play. Defaults to 100.
        hint (bool): whether to show the Basic Strategy (Thorp, 1966) as a hint. Defaults to False.

    Returns:
        A DescrStatsW object with the episodic reward distribution.

    Notes:
        The user can input either 0/1 or s/h (for stand/hit) as actions.
        If no valid input is entered, the default action is either 0 (if hint=False) or the basic strategy (if hint=True).
    """
    if hint:
        agent = BasicStrategyAgent(env)
    hist = defaultdict(int)
    for _ in range(episodes):
        total = 0.
        obs, reward, done = env.reset(), 0., False
        while True:
            print(env.render(), end=' ')
            if hint:
                a0 = agent.act(obs, reward, done)
                k = input(f'action: [hint: {action_labels[a0].lower()}] ')
            else:
                a0 = 0
                k = input(f'action: ')
            try:
                a = Action(int(k))
            except ValueError:
                try:
                    a = Action[action_labels.index(k.upper())]
                except KeyError:
                    a = a0
            obs, reward, done, _ = env.step(a)
            total += reward
            if done:
                hist[total] += 1
                print(env.render())
                break
    stats = ssw.DescrStatsW(
        data=list(hist.keys()),
        weights=list(hist.values())
    )
    return stats

