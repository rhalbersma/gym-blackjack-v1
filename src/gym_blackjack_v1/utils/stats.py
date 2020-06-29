#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import statsmodels.stats.weightstats as ssw

def descriptives(hist):
    return ssw.DescrStatsW(
        data=np.array(list(hist.keys())),
        weights=np.array(list(hist.values()), dtype=int)
    )
