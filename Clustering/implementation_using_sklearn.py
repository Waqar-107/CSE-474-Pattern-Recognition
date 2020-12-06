# from dust i have come, dust i will be

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys

sys.setrecursionlimit(10000)


class Solution:
    def __init__(self, dataset_file, k_nearest):
        self.k_nearest = k_nearest
        self.eps = 0
        self.min_pts = k_nearest
        self.dataset = []

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

        self.dataset = np.array(self.dataset)
        self.dataset /= np.array([mx, my])

        self.cluster = [0] * len(self.dataset)
        self.vis = [False] * len(self.dataset)
        self.colors = ['#585d8a', '#858482', '#23ccc9', '#e31712', '#91f881', '#89b84f', '#fedb00', '#0527f9', '#571d08',
                       '#ffae00', '#b31d5b', '#702d75']

    @staticmethod
    def euclidean_distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def estimate_eps(self):
        neighbors = NearestNeighbors(n_neighbors=self.k_nearest)
        neighbors_fit = neighbors.fit(self.dataset)
        distances, indices = neighbors_fit.kneighbors(self.dataset)

        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
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
        plt.figure(2)
        for i in range(len(self.dataset)):
            if self.cluster[i]:
                c = (self.cluster[i] - 1) % len(self.colors)
                c = self.colors[c]

                plt.scatter(self.dataset[i][0], self.dataset[i][1], color=c)

        plt.show()


np.random.seed(118)
solve = Solution("./data/blobs.txt", 4)
solve.estimate_eps()
solve.run_dbscan()

"""
bisecting - eps: 0.025
blob - eps: 0.06
moon - eps: 0.05

https://towardsdatascience.com/k-means-vs-dbscan-clustering-49f8e627de27
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e
"""
