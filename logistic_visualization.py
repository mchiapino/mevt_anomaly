import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import logistic as lg

# Fixing random state for reproducibility
np.random.seed(19680801)

d = 2
alphas = [[0], [1]]
as_dep = 0.15
n = int(1e3)
x_lgtc = lg.asym_logistic(d, alphas, n, as_dep)
x_gauss = abs(np.random.normal(scale=3.5, size=(n, 2)))

# Remove the extremer points
k_ = 20
ind_ = np.argsort(np.sum(x_lgtc, axis=1))[::-1]
x_lgtc = x_lgtc[ind_[k_:]]
# plt.plot(x_lgtc[:, 0], x_lgtc[:, 1], 'ob')
# k_extr = 200
# x_extr = x_lgtc[ind_extr[:k_extr]]
# plt.plot(x_extr[:, 0], x_extr[:, 1], 'or')
# plt.show()

# extreme points
k_extr = 35
ind_extr = np.argsort(np.sum(x_lgtc, axis=1))[::-1]
x_extr = x_lgtc[ind_extr[:k_extr]]

# the random data
x = x_lgtc[:, 0]
y = x_lgtc[:, 1]

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
# rect_histx = [left, bottom_h, width, 0.2]
# rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(10, 10))

axScatter = plt.axes(rect_scatter)
# axHistx = plt.axes(rect_histx)
# axHisty = plt.axes(rect_histy)

# # no labels
# axHistx.xaxis.set_major_formatter(nullfmt)
# axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x_gauss[:, 0], x_gauss[:, 1], color='b')
axScatter.scatter(x, y, color='b')
axScatter.scatter(x_extr[:, 0], x_extr[:, 1], color='r')

# now determine nice limits by hand:
binwidth = 0.25
xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((0, lim))
axScatter.set_ylim((0, lim))

# bins = np.arange(-lim, lim + binwidth, binwidth)
# axHistx.hist(x, bins=bins)
# axHisty.hist(y, bins=bins, orientation='horizontal')

# axHistx.set_xlim(axScatter.get_xlim())
# axHisty.set_ylim(axScatter.get_ylim())

plt.show()
