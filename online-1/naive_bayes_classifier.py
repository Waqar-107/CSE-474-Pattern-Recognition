# from dust i have come, dust i will be

import numpy as np
import math

number_of_features = 0
number_of_classes = 0
dataset_size = 0


class Object:
    def __init__(self, class_name):
        self.class_name = class_name
        self.features = []
        self.mean = []
        self.sd = []

    def get_mean(self, feature_no):
        sz = len(self.features)
        temp = np.array(self.features[: sz][feature_no]).astype(float)

        return np.mean(temp)

    def get_standard_deviation(self, feature_no):
        sz = len(self.features)
        temp = np.array(self.features[: sz][feature_no]).astype(float)

        return np.std(temp)

    def calc_mean(self):
        global number_of_features
        for i in range(number_of_features):
            mu = self.get_mean(i)
            self.mean.append(mu)

    def calc_standard_deviation(self):
        global number_of_features
        for i in range(number_of_features):
            sigma = self.get_standard_deviation(i)
            self.sd.append(sigma)


Object_Dictionary = {}


def read_dataset():
    global number_of_features, number_of_classes, dataset_size, Object_Dictionary

    f = open("./during coding/Train.txt", "r")
    lines = f.readlines()
    f.close()

    number_of_features, number_of_classes, dataset_size = map(int, lines[0].split())

    for i in range(dataset_size):
        data = lines[i + 1].split()

        class_name = data[number_of_features]

        if class_name not in Object_Dictionary:
            Object_Dictionary[class_name] = Object(class_name)

        Object_Dictionary[class_name].features.append(data[: number_of_features])


def train():
    global number_of_classes, number_of_features, Object_Dictionary, dataset_size

    for key in Object_Dictionary:
        obj = Object_Dictionary[key]
        obj.calc_mean()
        obj.calc_standard_deviation()


correct = 0


def test_accuracy():
    global number_of_features, Object_Dictionary, dataset_size, correct

    f = open("./during coding/Test.txt", "r")
    lines = f.readlines()
    f.close()

    wr = open("Report_coding.txt", "w")

    sample = 0
    for line in lines:
        sample += 1

        arr = []
        temp = line.rstrip()
        temp = temp.split()

        for i in range(len(temp)):
            t = temp[i].strip()
            if len(t):
                arr.append(float(t))

        class_name = temp[number_of_features]

        mx = 0
        predicted_class = -1
        for key in Object_Dictionary:
            obj = Object_Dictionary[key]

            mult = len(obj.features) / dataset_size
            for i in range(number_of_features):
                det = math.sqrt(2 * math.pi * obj.sd[i] * obj.sd[i])

                p = (arr[i] - obj.mean[i]) * (arr[i] - obj.mean[i])
                q = 2 * obj.sd[i] * obj.sd[i]

                x = (1 / det) * math.exp(-(p / q))

                mult *= x

            if mx < mult:
                predicted_class = obj.class_name
                mx = mult

        if predicted_class == class_name:
            correct += 1
        else:
            wr.write(str(sample) + " " + str(arr) + " " + str(class_name) + " "+ str(predicted_class) + "\n")

    acc = (correct / dataset_size) * 100
    wr.write("accuracy : " + str(acc))
    wr.close()


if __name__ == "__main__":
    read_dataset()
    train()
    test_accuracy()
