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
        self.co_variance = None

    def get_mean(self, feature_no):
        temp = np.array([i[feature_no] for i in self.features]).astype(float)
        return np.mean(temp)

    def get_standard_deviation(self, feature_no):
        temp = np.array([i[feature_no] for i in self.features]).astype(float)
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

    def calc_co_variance(self):
        global number_of_features, Object_Dictionary, dataset_size, number_of_classes
        self.co_variance = [[0 for j in range(number_of_features)] for i in range(number_of_features)]

        for i in range(number_of_features):
            for j in range(number_of_features):

                summation = 0.0
                # (Xi - mean of X) * (Yi - mean of Y) => Co-variance of x,y
                for k in range(len(self.features)):
                    x = self.features[k][i]
                    y = self.features[k][j]
                    x_mean = self.mean[i]
                    y_mean = self.mean[j]

                    summation += ((x - x_mean) * (x - x_mean) * (y - y_mean) * (y - y_mean))

                self.co_variance[i][j] = summation / len(self.features)


Object_Dictionary = {}


def read_dataset():
    global number_of_features, number_of_classes, dataset_size, Object_Dictionary

    f = open("./during_evaluation/Train.txt", "r")
    lines = f.readlines()
    f.close()

    number_of_features, number_of_classes, dataset_size = map(int, lines[0].split())

    for i in range(dataset_size):
        data = list(map(float, lines[i + 1].split()))

        class_name = data[number_of_features]

        if class_name not in Object_Dictionary:
            Object_Dictionary[class_name] = Object(class_name)

        Object_Dictionary[class_name].features.append(data[: number_of_features])


def train():
    global number_of_classes, number_of_features, Object_Dictionary, dataset_size

    for key in Object_Dictionary:
        obj = Object_Dictionary[key]
        obj.calc_mean()
        obj.calc_co_variance()


correct = 0


def test_accuracy():
    global number_of_features, Object_Dictionary, dataset_size, correct

    f = open("./during_evaluation/Test.txt", "r")
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
        predicted_class = ''
        for key in Object_Dictionary:
            obj = Object_Dictionary[key]

            mult = len(obj.features) / dataset_size

            # -------------------------------------------------------------
            # change here for the online
            # co-variance would be a square matrix
            co_variance = np.array(obj.co_variance)
            # print("class :", key, co_variance, np.linalg.det(co_variance))

            down = math.pow(2 * math.pi, number_of_features / 2) * math.sqrt(np.linalg.det(co_variance))

            # F - mu
            mat = []
            for i in range(number_of_features):
                mat.append(arr[i] - obj.mean[i])

            mat = np.array(mat)
            mat2 = mat
            mat = np.matrix.transpose(mat)

            temp = mat.dot(np.linalg.inv(co_variance))
            temp = temp.dot(mat2)

            up = -0.5 * temp
            mult = math.exp(up) / down
            # -------------------------------------------------------------

            if mx <= mult:
                predicted_class = obj.class_name
                mx = mult

        if int(predicted_class) == int(class_name):
            correct += 1
        else:
            wr.write("sample no: " + str(sample) + ", feat:" + str(arr[: number_of_features]) \
                     + ", actual-class:" + str(class_name) + ", predicted-class: " + str(predicted_class) + "\n")

    acc = (correct / dataset_size) * 100
    wr.write("accuracy : " + str(acc))
    wr.close()


if __name__ == "__main__":
    read_dataset()

    train()
    test_accuracy()


'''
// dot
A = np.array([[], [], []])
B = np.array([[], [], []])
A.dot(B)


// inverse
X = np.linalg.inv(A)

// determinant
np.linalg.det(A)

// ln
math.log(val, math.e)
'''


