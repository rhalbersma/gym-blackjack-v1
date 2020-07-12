#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import gym
from ..agents import BasicStrategyAgent
from ..envs import Action

def summary(episodes, total):
    print(f"""
============================================================
Episodes: {episodes:>4}, total reward: {total:>+5.1f}, average reward: {total / episodes:>+7.1%}
============================================================
"""
    )

def play(env, episodes=100, hint=False):
    """
    Play an interactive session of blackjack.

    Args:
        env (object): an OpenAI Gym blackjack environment, either 'Blackjack-v0' or 'Blackjack-v1'.
        episodes (int): the number of episodes to play. Defaults to 100.
        hint (bool): whether to show the Basic Strategy (Thorp, 1966) as a hint. Defaults to False.

    Returns:
        The average reward per episode.

    Notes:
        The user can input either 0/1 or s/h (for stand/hit) as actions.
        If no valid input is entered, the default action is either 0 (if hint=False) or the basic strategy (if hint=True).
    """
    if hint:
        agent = BasicStrategyAgent(env)
    total = 0.
    for i, _ in enumerate(range(episodes)):
        obs, reward, done = env.reset(), 0., False
        while True:
            print(env.render(), end=' ')
            if hint:
                a0 = agent.act(obs, reward, done)
                k = input(f'action: [hint: {Action(a0).name}] ')
            else:
                a0 = 0
                k = input(f'action: ')
            try:
                a = Action(int(k))  # try 0/1
            except ValueError:
                try:
                    a = Action[k]   # try s/h
                except KeyError:
                    a = a0          # fallback to the Basic Strategy
            obs, reward, done, _ = env.step(a)
            total += reward
            if done:
                break
        print(env.render())
        summary(i + 1, total)
    return total / episodes
