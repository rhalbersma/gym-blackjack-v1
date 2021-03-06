#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from ..enums import Action


class AlwaysHitAgent:
    """
    An agent that always hits.
    """
    def __init__(self, env=None):
        pass

    def act(self, obs, reward, done):
        return Action.HIT

