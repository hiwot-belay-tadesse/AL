import numpy as np 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


X = np.random.normal(0, 10, (71, 64))  
X_prime = np.random.uniform(0, 10, (71, 64))  

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
pred = kmeans.predict(X_prime)

centeriods = kmeans.cluster_centers_
tsne = TSNE(n_components=2, random_state=0)

X_combined = np.vstack((X, centeriods))
combined_tsne = tsne.fit_transform(X_combined)
X_tsne = combined_tsne[:len(X)]
centeriods_tsne = combined_tsne[len(X):]

fig, ax = plt.subplots()s
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=pred, cmap='viridis')
ax.scatter(centeriods_tsne[:, 0], centeriods_tsne[:, 1], c='red', marker='X')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

def split_df_w_k_center(z,n_clusters):
    Z = 
    