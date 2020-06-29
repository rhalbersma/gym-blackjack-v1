#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from collections import defaultdict
from tqdm import tqdm

def simulate(agent, env, episodes=10**6):
    """
    Simulate an agent in an environment over a number of episodes.

    Args:
        agent (object): an agent with an act(obs, reward, done) method.
        env (object): an OpenAI Gym environment.
        episodes (int): the number of episodes to simulate (default: 1,000,000).

    Returns:
        A dictionary with the episodic reward distribution.

    Notes:
        A progress bar is shown.
    """
    hist = defaultdict(int)
    for _ in tqdm(range(episodes)):
        total = 0.
        obs, reward, done = env.reset(), 0., False
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            total += reward
            if done:
                hist[total] += 1
                break
    return hist
