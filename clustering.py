##K-MEANS

#The seed() function initializes the random number generator with a specified seed value. When you set a seed value, you ensure that the sequence of random numbers
#generated is reproducible. In other words, every time you run the code with the same seed, you'll get the same sequence of random numbers.
#Here, X is initialized as a NumPy array of shape (50, 2). It contains random numbers sampled from a standard normal distribution
#(mean = 0, standard deviation = 1).
np.random.seed(0)
x = np.random.standard_normal((50, 2))
# print(x)
x[:25, 0] += 3
x[:25, 1] -= 4
# print(x)
kmeans = KMeans(n_clusters=2, random_state=2, n_init=20).fit(x)
a = kmeans.labels_
# print(a)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(x[:, 0], x[:, 1], c=a)
ax.set_title("k-means clustering results with k=2")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=3, n_init=20).fit(x)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x[:, 0], x[:, 1], c=kmeans.labels_)
ax.set_title("k-means clustering results with k=3")
plt.show()

kmeans1 = KMeans(n_clusters=3, random_state=3, n_init=1).fit(x)
kmeans20 = KMeans(n_clusters=3, random_state=3, n_init=20).fit(x)
b = kmeans1.inertia_, kmeans20.inertia_
print(b)








###Hierarchical clustering
np.random.seed(0)
x = np.random.standard_normal((50, 2))
# print(x)
x[:25, 0] += 3
x[:25, 1] -= 4
# print(x)
hclust = AgglomerativeClustering
hc_comp = hclust(distance_threshold=0, n_clusters=None, linkage='complete')
hc_comp.fit(x)
hc_av = hclust(distance_threshold=0, n_clusters=None, linkage='average')
hc_av.fit(x)
hc_sing = hclust(distance_threshold=0, n_clusters=None, linkage='single')
hc_sing.fit(x)
d = np.zeros((x.shape[0], x.shape[0]))
for i in range((x.shape[0])):
    x_ = np.multiply.outer(np.ones(x.shape[0]), x[i])
    d[i] = np.sqrt(np.sum((x-x_)**2, 1))
hc_sing_pre = hclust(distance_threshold=0, n_clusters=None, metric='precomputed', linkage='single')
hc_sing_pre.fit(d)
cargs = {'color_threshold':-np.inf, 'above_threshold_color': 'black'}
linkage_comp = compute_linkage((hc_comp))
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp, ax=ax, **cargs)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp, ax=ax, color_threshold=4, above_threshold_color='black')
plt.show()
c = cut_tree(linkage_comp, n_clusters=4).T
# print(c)
d = cut_tree(linkage_comp, height=5)
# print(d)
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
hc_comp_scale = hclust(distance_threshold=0, n_clusters=None, linkage='complete').fit(x_scale)
linkage_comp_scale = compute_linkage(hc_comp_scale)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp_scale, ax=ax, **cargs)
ax.set_title("hierarchical clustering with scaled features")
plt.show()
x = np.random.standard_normal((30, 3))
corD = 1 - np.corrcoef(x)
hc_cor = hclust(linkage='complete', distance_threshold=0, n_clusters=None, metric='precomputed')
hc_cor.fit(corD)
linkage_cor = compute_linkage(hc_cor)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_cor, ax=ax, **cargs)
ax.set_title("complete linkage with correlation-based dissimilarity")
plt.show()






