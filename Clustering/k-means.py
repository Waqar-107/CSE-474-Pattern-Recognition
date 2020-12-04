# from dust i have come, dust i will be

import matplotlib.pyplot as plt
import numpy as np


class Solution:
    def __init__(self, dataset_file, k_nearest):
        self.dataset = []
        self.k_nearest = k_nearest
        self.eps = 0.0
        self.min_pts = 0.0

        file = open(dataset_file, "r")
        lines = file.readlines()

        for line in lines:
            if line[-1] == "\n":
                line = line[:-1]

            x, y = map(float, line.split())
            self.dataset.append([x, y])

        self.dataset = np.array(self.dataset)

    @staticmethod
    def euclidean_distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def draw_plot_of_k_th_nearest_neighbor(self):
        dist = []
        for i in range(len(self.dataset)):
            temp = []
            for j in range(len(self.dataset)):
                if i != j:
                    temp.append(self.euclidean_distance(self.dataset[i], self.dataset[j]))

            temp.sort()
            dist.append(temp[self.k_nearest - 1])

        X = [i for i in range(len(self.dataset))]

        dist.sort()
        for i in range(len(X)):
            print(X[i], dist[i])

        plt.figure(1)
        plt.plot(X, dist)
        plt.grid()
        plt.show()

        # self.eps = float(input("EPS?"))
        # self.min_pts = float(input("MINPTS?"))
        self.eps = 0.3
        self.min_pts = int(np.log(len(self.dataset)))


solve = Solution("./data/blobs.txt", 4)
solve.draw_plot_of_k_th_nearest_neighbor()

