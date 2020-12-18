# from dust i have come, dust i will be

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
from copy import deepcopy

sys.setrecursionlimit(10000)
max_itr = 1000


class Solution:
    def __init__(self, dataset_file, k_nearest):
        self.k_nearest = k_nearest
        self.eps = 0
        self.min_pts = k_nearest
        self.dataset = []
        self.number_of_cluster = 0

        file = open(dataset_file, "r")
        lines = file.readlines()

        mx, my = -np.inf, -np.inf
        for line in lines:
            if line[-1] == "\n":
                line = line[:-1]

            x, y = map(float, line.split())
            mx = max(abs(x), mx)
            my = max(abs(y), my)

            self.dataset.append([x, y])

        self.dataset.sort(key=lambda xy: (xy[0], xy[1]))

        self.dataset = np.array(self.dataset)
        self.dataset /= np.array([mx, my])

        self.cluster = [0] * len(self.dataset)
        self.vis = [False] * len(self.dataset)
        self.colors = ['#585d8a', '#858482', '#23ccc9', '#e31712', '#91f881', '#89b84f', '#fedb00', '#0527f9',
                       '#571d08', '#ffae00', '#b31d5b', '#702d75']

    @staticmethod
    def euclidean_distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # uses NearestNeighbors from sklearn
    # first we tell it the value of k, then we feed it with data
    # after that it returns distances of first k nearest neighbors.
    # we can explicitly tell it to use particular data structures too(ball-tree, kd-tree etc.)
    def estimate_eps(self):
        neighbors = NearestNeighbors(n_neighbors=self.k_nearest)
        neighbors_fit = neighbors.fit(self.dataset)
        distances, indices = neighbors_fit.kneighbors(self.dataset)

        distances = np.sort(distances, axis=0)
        distances = distances[:, self.k_nearest - 1]
        plt.plot(distances, color='#23ccc9')
        plt.grid()
        plt.show()

        # from the plot
        self.eps = float(input("what is the estimated eps from the plot?"))

    def run_dbscan_util(self, src, c):
        self.cluster[src] = c
        self.vis[src] = True

        for i in range(len(self.dataset)):
            if not self.vis[i] and self.euclidean_distance(self.dataset[src], self.dataset[i]) <= self.eps:
                self.run_dbscan_util(i, c)

    def run_dbscan(self):
        core_points = []

        for i in range(len(self.dataset)):
            neighbors = 0
            for j in range(len(self.dataset)):
                if i != j:
                    if self.euclidean_distance(self.dataset[i], self.dataset[j]) <= self.eps:
                        neighbors += 1

            if neighbors >= self.min_pts:
                core_points.append(i)

        c = 0
        self.vis = [False] * len(self.dataset)
        np.random.shuffle(core_points)

        for p in core_points:
            if self.vis[p]:
                continue

            c += 1
            self.run_dbscan_util(p, c)

        print("total clusters formed:", c)
        self.number_of_cluster = c

        plt.figure(2)
        for i in range(len(self.dataset)):
            if self.cluster[i]:
                c = (self.cluster[i] - 1) % len(self.colors)
                c = self.colors[c]

                plt.scatter(self.dataset[i][0], self.dataset[i][1], color=c)

        plt.show()

    def k_means(self):
        centroid_idx = []
        interval = len(self.dataset) // self.number_of_cluster
        for i in range(self.number_of_cluster):
            centroid_idx.append((interval * i))

        centroids = []
        for i in range(self.number_of_cluster):
            centroids.append(self.dataset[centroid_idx[i]])

        temp_clusters = [-1] * len(self.dataset)
        dist = [np.inf] * len(self.dataset)

        for itr in range(max_itr):
            print("iteration -", itr)

            # for all find the closest centroid
            for i in range(len(self.dataset)):
                dist[i] = np.inf
                for j in range(self.number_of_cluster):
                    d = self.euclidean_distance(centroids[j], self.dataset[i])
                    if d < dist[i]:
                        dist[i] = d
                        temp_clusters[i] = j

            # find out new centroids
            clusters = [[] for _ in range(self.number_of_cluster)]
            for i in range(len(self.dataset)):
                clusters[temp_clusters[i]].append(self.dataset[i])

            new_centroids = []
            for i in range(self.number_of_cluster):
                c = np.array(clusters[i])
                mean = np.average(c, axis=0)
                new_centroids.append(mean)

            flag = True
            for i in range(self.number_of_cluster):
                a = np.abs(centroids[i] - new_centroids[i])
                if (a != 0).any():
                    flag = False

            if flag:
                break

            centroids = deepcopy(new_centroids)

        plt.figure(3)
        for i in range(len(self.dataset)):
            if temp_clusters[i] >= 0:
                c = temp_clusters[i] % len(self.colors)
                c = self.colors[c]

                plt.scatter(self.dataset[i][0], self.dataset[i][1], color=c)

        for c in centroids:
            plt.scatter(c[0], c[1], color='#000000', marker='p', linewidths=5)
        plt.show()


np.random.seed(118)
solve = Solution("./data/moons.txt", 4)
solve.estimate_eps()
solve.run_dbscan()
solve.k_means()

"""
bisecting - eps: 0.03
blob - eps: 0.08
moon - eps: 0.06

https://towardsdatascience.com/k-means-vs-dbscan-clustering-49f8e627de27
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
https://towardsdatascience.com/clustering-using-k-means-algorithm-81da00f156f6
"""
