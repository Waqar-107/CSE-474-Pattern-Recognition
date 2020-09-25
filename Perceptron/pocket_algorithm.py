import numpy as np
import sys

number_of_features = 0
number_of_classes = 0
dataset_size = 0

dataset = []
class_names = []

max_itr = 1000
seed_val = 107
np.random.seed(seed_val)


def read_dataset(train_file):
    global number_of_features, number_of_classes, dataset_size, dataset, class_names

    f = open(train_file, "r")
    lines = f.readlines()

    number_of_features, number_of_classes, dataset_size = map(int, lines[0].rstrip().split())

    for i in range(dataset_size):
        data = lines[i + 1].rstrip().split()
        dataset.append(np.array(data[: number_of_features], dtype=float))
        class_names.append(int(data[number_of_features]))


class Pocket_Perceptron:
    def __init__(self):
        self.w = np.random.uniform(-1, 1, number_of_features + 1)
        self.ws = self.w
        self.hs = 0
        self.learning_rate = 0.1

    def train_model(self):
        global dataset, dataset_size, class_names, max_itr

        for itr in range(max_itr):
            misclassified = []
            for i in range(dataset_size):
                x = np.array(dataset[i])
                x = np.append(x, 1)
                actual_class = class_names[i]

                prod = np.dot(self.w, x)

                if actual_class == 1 and prod < 0:
                    misclassified.append(x * -1)
                elif actual_class == 2 and prod >= 0:
                    misclassified.append(x)

            if self.hs < dataset_size - len(misclassified):
                self.hs = dataset_size - len(misclassified)
                self.ws = self.w

            # all got classified
            if len(misclassified) == 0:
                sys.stdout.write("training done at " + str(itr + 1) + "th iteration\n")
                sys.stdout.write("w: " + str(self.ws) + "\n")
                break

            # update w
            summation = sum(misclassified)
            self.w = self.w - self.learning_rate * summation

    def test_model(self, test_file):
        global class_names

        correctly_classified = 0
        results = open("results.txt", "w")
        results.write("Pocket Algorithm\n\n")

        f = open(test_file, "r")
        lines = f.readlines()

        for i in range(len(lines)):
            data = list(map(float, lines[i].rstrip().split()))
            actual_class = int(data[number_of_features])
            data[number_of_features] = 1
            data = np.array(data)

            prod = np.dot(self.ws, data)

            if prod > 0:
                predicted_class = 1
            else:
                predicted_class = 2

            if actual_class == predicted_class:
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
    p = Pocket_Perceptron()
    p.train_model()
    p.test_model("./dataset/testLinearlyNonSeparable.txt")
