import numpy as np
import copy


class Features:
    def __init__(self):
        self.number_of_classes = 0
        self.x = []
        self.y = []
        self.class_set = set()
        self.class_no = []
        self.num_of_features = None
        self.num_of_sample = 0

    def count_num_of_sample(self, filename):
        file = open(filename, "r")
        line = file.readline().split()
        self.num_of_features = int(len(line)) - 1
        while len(line) != 0:
            self.num_of_sample += 1
            self.class_set.add(int(line[self.num_of_features]))
            line = file.readline().split()
        self.number_of_classes = len(self.class_set)
        file.close()

    def read_feature_value(self, filename):
        file = open(filename, "r")
        for i in range(self.num_of_sample):
            x = np.zeros((self.num_of_features, 1))
            y = np.zeros((self.number_of_classes, 1))
            line = file.readline().split()
            for j in range(self.num_of_features):
                x[j][0] = float(line[j])
            y[int(line[self.num_of_features]) - 1][0] = 1.0
            self.x.append(x)
            self.y.append(y)
            self.class_no.append(int(line[self.num_of_features]))
        file.close()

    def normalize_input(self):
        self.x = np.array(self.x)
        self.x = (self.x - self.x.mean(axis=0)) / self.x.std(axis=0)


class Backpropagation:
    def __init__(self, num_of_layer, num_of_node_in_a_hidden_layer, features):
        self.num_of_layer = num_of_layer
        self.y_hat = []
        self.w = []
        self.b = []
        self.v = []
        self.fx_dash = []
        self.num_of_node = num_of_node_in_a_hidden_layer
        self.mu = 0.01
        self.features = features

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_diff(self, fx):
        return fx * (1 - fx)

    def set_random_weight(self):
        np.random.seed(1)
        for i in range(self.num_of_layer):
            if i == 0:
                w = np.random.randn(self.num_of_node, self.features.num_of_features)
                self.w.append(w)
            elif i == self.num_of_layer - 1:
                w = np.random.randn(self.features.number_of_classes, self.num_of_node)
                self.w.append(w)
            else:
                w = np.random.randn(self.num_of_node, self.num_of_node)
                self.w.append(w)
        self.b = np.zeros((1, self.num_of_layer))

    def calculate_del_for_layer_L(self, sample_no, layer_no):
        del_L = np.subtract(self.y_hat[sample_no][layer_no], self.features.y[sample_no])
        del_L = np.multiply(del_L, self.fx_dash[sample_no][layer_no])
        return del_L

    def calculate_del_for_layer_less_than_L(self, prev_del, sample_no, layer_no):
        del_r = np.dot(self.w[layer_no], np.diagflat(self.fx_dash[sample_no][layer_no]))
        del_r = np.dot(prev_del.T, del_r)
        return del_r.T

    def predict(self, y_hat, actual_class):
        class_value = 0
        for i in range(self.features.number_of_classes):
            if y_hat[class_value][0] < y_hat[i][0]:
                class_value = i
        if actual_class == class_value + 1:
            return True
        else:
            return False

    def calculate_error(self, y_hat, y):
        err = np.subtract(y_hat, y)
        err = np.square(err)
        return np.sum(err)

    def forward_propagation(self):
        error = 0.0
        no_of_success = 0
        for i in range(self.features.num_of_sample):
            v = []
            y = [self.features.x[i]]
            fx_dash = [self.sigmoid_diff(self.features.x[i])]
            for j in range(self.num_of_layer):
                v.append(np.dot(self.w[j], y[j]) + self.b[0][j])
                y.append(self.sigmoid(np.dot(self.w[j], y[j]) + self.b[0][j]))
                fx_dash.append(self.sigmoid_diff(y[j + 1]))
            if self.predict(y[self.num_of_layer], self.features.class_no[i]):
                no_of_success += 1
            error += self.calculate_error(y[self.num_of_layer], self.features.y[i])
            self.v.append(v)
            self.y_hat.append(y)
            self.fx_dash.append(fx_dash)
        print(error / 2.0)

    def back_propagation(self):
        del_r = None
        w_new = copy.deepcopy(self.w)
        for i in range(self.features.num_of_sample):
            for j in range(self.num_of_layer, 0, -1):
                if j == self.num_of_layer:
                    del_r = self.calculate_del_for_layer_L(i, j)
                else:
                    del_r = self.calculate_del_for_layer_less_than_L(del_r, i, j)

                # print(del_r)
                # print(self.y_hat[i][j - 1].T)
                # print(np.dot(del_r, self.y_hat[i][j - 1].T))

                w_new[j - 1] -= self.mu * np.dot(del_r, self.y_hat[i][j - 1].T)
        self.w = w_new

    def train(self):
        itr = 1000
        for i in range(itr):
            print(i + 1, end=" ")
            self.forward_propagation()
            self.back_propagation()
            self.v.clear()
            self.y_hat.clear()
            self.fx_dash.clear()

    def test(self, test_feature):
        num_of_success = 0
        error = 0
        print("params")
        print(self.w)
        for i in range(test_feature.num_of_sample):
            y = [test_feature.x[i]]
            for j in range(self.num_of_layer):
                y.append(self.sigmoid(np.dot(self.w[j], y[j]) + self.b[0][j]))
            if self.predict(y[self.num_of_layer], test_feature.class_no[i]):
                num_of_success += 1
            error += self.calculate_error(y[self.num_of_layer], test_feature.y[i])
        print("Test Accuracy: ", num_of_success, error / 2.0)


if __name__ == '__main__':
    trainFileName = "./dataset/trainNN.txt"
    testFileName = "./dataset/testNN.txt"

    feature = Features()
    feature.count_num_of_sample(trainFileName)
    feature.read_feature_value(trainFileName)
    feature.normalize_input()

    testFeature = Features()
    testFeature.count_num_of_sample(testFileName)
    testFeature.read_feature_value(testFileName)
    testFeature.normalize_input()

    backprop = Backpropagation(4, 3, feature)
    backprop.set_random_weight()
    backprop.train()
    backprop.test(testFeature)
