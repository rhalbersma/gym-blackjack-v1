#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from collections import defaultdict

import statsmodels.stats.weightstats as ssw
from tqdm import tqdm


def simulate(agent, env, start=None, episodes=1_000_000):
    """
    Simulate an agent in an environment over a number of episodes.

    Args:
        agent (object): an agent with an act(obs, reward, done) method.
        env (object): an OpenAI Gym environment.
        episodes (int): the number of episodes to simulate. Defaults to 1,000,000.

    Returns:
        A DescrStatsW object with the episodic reward distribution.

    Notes:
        A progress bar is shown.
    """
    hist = defaultdict(int)
    for _ in tqdm(range(episodes)):
        total = 0.
        obs = env.reset() if start is None else env.explore(start)
        reward, done = 0., False
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            total += reward
            if done:
                hist[total] += 1
                break
    stats = ssw.DescrStatsW(
        data=list(hist.keys()),
        weights=list(hist.values())
    )
    return stats

