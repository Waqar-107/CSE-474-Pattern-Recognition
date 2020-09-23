# pocket algorithm

import numpy as np

number_of_features = 0
number_of_classes = 0
dataset_size = 0
w = None
max_itr = 1000

test_file = "./dataset/testLinearlyNonSeparable.txt"
train_file = "./dataset/trainLinearlyNonSeparable.txt"


class Object:
    def __init__(self, class_name):
        self.class_name = class_name
        self.features = []


Object_Dictionary = {}


def read_dataset():
    global number_of_features, number_of_classes, dataset_size, Object_Dictionary

    f = open(train_file, "r")
    lines = f.readlines()

    number_of_features, number_of_classes, dataset_size = map(int, lines[0].rstrip().split())

    for i in range(dataset_size):
        data = lines[i + 1].rstrip().split()
        class_name = int(data[number_of_features])

        if class_name not in Object_Dictionary:
            Object_Dictionary[class_name] = Object(class_name)

        Object_Dictionary[class_name].features.append(np.array(data[: number_of_features], dtype=float))


def train_model():
    global w

    np.random.seed(107)
    w = np.random.uniform(-1, 1, number_of_features + 1)

    learning_rate = 0.1

    for itr in range(max_itr):
        misclassified = []

        for key in Object_Dictionary.keys():
            for i in range(len(Object_Dictionary[key].features)):
                x = Object_Dictionary[key].features[i]
                x = np.append(x, 1)
                x = np.array(x)

                val = np.dot(w, x)

                # actually omega1, classified as omega2
                if key == 1 and val < 0:
                    misclassified.append(x * -1)

                # actually omega2, classified as omega1
                elif key == 2 and val > 0:
                    misclassified.append(x)

        if len(misclassified) == 0:
            print("training done in", itr, "th iteration")
            break

        summation = np.zeros(number_of_features + 1)
        for i in range(len(misclassified)):
            summation += misclassified[i]

        summation = learning_rate * summation
        w = w - summation


def test_model():
    global w
    correctly_detected = 0

    f = open(test_file, "r")
    lines = f.readlines()

    for i in range(dataset_size):
        data = list(map(float, lines[i].rstrip().split()))

        actual_class = int(data[number_of_features])
        data[number_of_features] = 1
        x = np.array(data)

        prod = np.dot(w, x)
        if prod >= 0:
            predicted_class = 1
        else:
            predicted_class = 2

        if predicted_class == actual_class:
            correctly_detected += 1

    accuracy = (correctly_detected / dataset_size) * 100
    print("accuracy :", accuracy, "%")


if __name__ == "__main__":
    read_dataset()
    train_model()
    test_model()
