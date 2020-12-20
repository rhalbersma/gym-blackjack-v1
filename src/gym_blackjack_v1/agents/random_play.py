#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)


class RandomPlayAgent:
    """
    An agent that randomly samples an action.
    """
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs, reward, done):
        return self.action_space.sample()

