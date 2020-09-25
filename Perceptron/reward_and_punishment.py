# reward and punishment algorithm

import numpy as np
import sys

number_of_features = 0
number_of_classes = 0
dataset_size = 0

dataset = []
classname = []

max_itr = 1000


def read_dataset(train_file):
    global number_of_features, number_of_classes, dataset_size, dataset

    f = open(train_file, "r")
    lines = f.readlines()

    number_of_features, number_of_classes, dataset_size = map(int, lines[0].rstrip().split())

    for i in range(dataset_size):
        data = lines[i + 1].rstrip().split()
        dataset.append(np.array(data, dtype=float))
        classname.append(int(data[number_of_features]))


class Reward_and_Punishment:
    def __init__(self):
        self.w = np.zeros(number_of_features + 1)
        self.learning_rate = 0.1

    def train_model(self):
        global dataset, classname

        for itr in range(max_itr):
            flag = True
            for i in range(dataset_size):
                x = dataset[i]
                actual_class = classname[i]
                x[number_of_features] = 1.0

                val = np.dot(self.w, x)

                # actually omega1, classified as omega2
                if actual_class == 1 and val <= 0.0:
                    self.w = self.w + self.learning_rate * x
                    flag = False

                # actually omega2, classified as omega1
                elif actual_class == 2 and val >= 0.0:
                    self.w = self.w - self.learning_rate * x
                    flag = False

            if flag:
                print("stopping at", itr, "th iteration")
                print("weight vector", self.w)
                break

    def test_model(self, test_file):
        correctly_classified = 0

        results = open("./dataset/results.txt", "w")
        results.write("Reward and Punishment\n\n")

        f = open(test_file, "r")
        lines = f.readlines()

        for i in range(len(lines)):
            data = list(map(float, lines[i].rstrip().split()))

            actual_class = int(data[number_of_features])
            data[number_of_features] = 1
            x = np.array(data)

            prod = np.dot(self.w, x)
            if prod >= 0:
                predicted_class = 1
            else:
                predicted_class = 2

            if predicted_class == actual_class:
                correctly_classified += 1
            else:
                results.write("sample no.: " + str(i + 1) + ". feature value: " + str(
                    data[:number_of_features]) + ". actual class: " + str(actual_class) + ". predicted class: " + str(
                    predicted_class) + "\n")

        accuracy = (correctly_classified / len(lines)) * 100
        sys.stdout.write("Correctly classified: " + str(correctly_classified) + "\n")
        sys.stdout.write("Accuracy: " + str(accuracy))

        results.write("Accuracy: " + str(accuracy))


if __name__ == "__main__":
    read_dataset("./dataset/trainLinearlyNonSeparable.txt")
    p = Reward_and_Punishment()
    p.train_model()
    p.test_model("./dataset/testLinearlyNonSeparable.txt")
