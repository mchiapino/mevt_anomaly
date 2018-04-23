import numpy as np
import matplotlib.pyplot as plt

import logistic as lg

d = 2
alphas = [[0, 1], [0], [1]]
as_dep = 0.1
n = int(1e3)
x_lgtc = lg.asym_logistic(d, alphas, n, as_dep)
plt.plot(x_lgtc[:, 0], x_lgtc[:, 1], 'o')
plt.show()
