from sklearn.cluster import KMeans
import numpy as np


class ZonesExtractor:
    def __init__(self, n_clusters):
        # Assume X_train is your training data and X_test is your new data
        self.kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)

    def fit(self, X):
        self.kmeans.fit(X)
        # Calculate the maximum distance to centroid for each cluster in training data
        self.clusters_max_distances = []
        for i in range(self.kmeans.n_clusters):
            cluster_points = X[self.kmeans.labels_ == i]
            centroid = self.kmeans.cluster_centers_[i]
            distances = np.sqrt(((cluster_points - centroid) ** 2).sum(axis=1))
            self.clusters_max_distances.append(np.max(distances))

    def find_nearest_zone(self, X):
        # Predict clusters for new data
        labels = self.kmeans.predict(X)
        labels = labels.astype(float)
        new_clusters = []
        new_clusters_db_eps = []
        # Check if any point is farther from its centroid than the maximum distance in training data
        for i, label in enumerate(labels):
            label = int(label)
            centroid = self.kmeans.cluster_centers_[label]
            dist = np.sqrt(((X[i] - centroid) ** 2).sum())
            if dist > self.clusters_max_distances[label]:
                # Assign to a new cluster
                new_label = labels[i] + i / 100000
                print(labels[i], dist, self.clusters_max_distances[label], new_label)
                labels[i] = new_label
                if not (labels[i] in new_clusters):
                    new_clusters.append(labels[i])
                    new_clusters_db_eps.append(self.clusters_max_distances[label])

        return labels
