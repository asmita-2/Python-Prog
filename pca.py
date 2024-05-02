###this code performs pca is US-ARREST dataset.


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, cut_tree
from ISLP.cluster import compute_linkage

###loading data.
us_arrests = get_rdataset('USArrests').data
#print(us_arrests)
# print(us_arrests.columns)
# print(us_arrests.mean())
#print(us_arrests.var())

###standardizing the data.
scaler = StandardScaler(with_std=True, with_mean=True)
us_scaled = scaler.fit_transform(us_arrests)

##fitting the model
pca_us = PCA()
pca_us.fit(us_scaled)
#print(pca_us.mean_)
scores = pca_us.transform(us_scaled)

##printing scores and components of pca.
#print(scores)
#print(pca_us.components_)

### PCA visualisation bi-plot method
i, j = 0, 1
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:, 0], scores[:, 1])
ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))
for k in range(pca_us.components_.shape[1]):
    ax.arrow(0, 0, pca_us.components_[i, k], pca_us.components_[j, k])
    ax.text(pca_us.components_[i, k],
            pca_us.components_[j, k],
            us_arrests.columns[k])

##scaling
scale_arrow = s_ = 2
scores[:, 1] *= -1
pca_us.components_[1] *= -1
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:, 0], scores[:, 1])
ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))
for k in range(pca_us.components_.shape[1]):
    ax.arrow(0, 0, s_*pca_us.components_[i, k], s_*pca_us.components_[j, k])
    ax.text(s_*pca_us.components_[j, k],
            s_*pca_us.components_[j,k],
            us_arrests.columns[k])
a = scores.std(0, ddof=1)

# StdDev of Principal component scores
#print(a)

##Variance of each score
b = pca_us.explained_variance_ratio_
#print(b)

##Plotting Proportion of variance explained
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ticks = np.arange(pca_us.n_components_)+1
ax = axes[0]
ax.plot(ticks, pca_us.explained_variance_ratio_, marker='o')
ax.set_xlabel('principal component')
ax.set_ylabel('proportion of variance explained')
ax.set_ylim([0, 1])
ax.set_xticks(ticks)

###cumulative variance plot
ax = axes[1]
ax.plot(ticks,
pca_us.explained_variance_ratio_.cumsum(), marker='o')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Cumulative Proportion of Variance Explained')
ax.set_ylim([0, 1])
ax.set_xticks(ticks)
plt.show()
c = np.array([1, 2, 8, -3])
np.cumsum(c)
print(c)

