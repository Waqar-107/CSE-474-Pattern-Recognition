import numpy as np
import copy

seed_value = 1
np.random.seed(seed_value)

test_x = None
test_y = []
train_x = None
train_y = None

number_of_features = None
number_of_classes = None

test_file_name = "./dataset/testNN.txt"
train_file_name = "./dataset/trainNN.txt"

number_of_layer = None
nodes_in_each_layer = []

parameters = {}
max_itr = 1000
mu = 0.01


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def sigmoid_differentiated(val):
    return sigmoid(val) * (1 - sigmoid(val))


def scale_data():
    global train_x, test_x, number_of_features

    mean = train_x.mean(axis=1).reshape((number_of_features, 1))
    sd = train_x.std(axis=1).reshape((number_of_features, 1))
    train_x = (train_x - mean) / sd

    mean = test_x.mean(axis=1).reshape((number_of_features, 1))
    sd = test_x.std(axis=1).reshape((number_of_features, 1))
    test_x = (test_x - mean) / sd


def read_dataset():
    global number_of_features, number_of_classes, train_x, train_y, test_x, test_y, \
        train_file_name, test_file_name

    # read dataset to train
    train_file = open(train_file_name)
    lines = train_file.readlines()

    assert (len(lines) > 0)

    number_of_features = len(lines[0].strip().split()) - 1
    number_of_classes = 0

    features = []
    classes = []
    for line in lines:
        values = line.strip().split()
        features.append(values[:number_of_features])
        classes.append(int(values[number_of_features]))

        number_of_classes = max(number_of_classes, int(values[number_of_features]))

    train_x = np.array(features, dtype=float).T
    train_y = np.zeros((number_of_classes, len(lines)))
    for i in range(len(lines)):
        train_y[classes[i] - 1][i] = 1

    # train_x : (number of features, dataset size)
    # train_y : (number of classes, dataset size)
    assert (train_x.shape == (number_of_features, len(lines)))
    assert (train_y.shape == (number_of_classes, len(lines)))

    # read dataset to test
    test_file = open(test_file_name)
    lines = test_file.readlines()

    features = []
    for line in lines:
        values = line.strip().split()
        features.append(values[:number_of_features])
        test_y.append(int(values[number_of_features]))

    test_x = np.array(features, dtype=float).T

    # test_x : (number of features, dataset size)
    # train_y : simple 1D array
    assert (test_x.shape == (number_of_features, len(lines)))

    # free memories
    del features, classes, lines


def initialize_parameters(network_description):
    global number_of_layer, nodes_in_each_layer, number_of_classes, number_of_features, parameters
    parameters = {}

    # input layer -> hidden layers -> output layers
    number_of_layer = len(network_description) - 1
    nodes_in_each_layer = network_description

    for i in range(1, number_of_layer + 1, 1):
        parameters["W" + str(i)] = np.random.randn(nodes_in_each_layer[i], nodes_in_each_layer[i - 1])
        assert (parameters["W" + str(i)].shape == (nodes_in_each_layer[i], nodes_in_each_layer[i - 1]))


def forward_propagation(input_vector):
    global number_of_layer, parameters

    # Y: (features, dataset size)
    Y = np.array(input_vector)
    parameters["Y0"] = Y

    for i in range(1, number_of_layer + 1, 1):
        # Wi: (nodes in ith, nodes in i-1th)
        # Y: (nodes in i-1th, dataset size)
        # V: (nodes in ith, dataset size)
        V = np.dot(parameters["W" + str(i)], Y)
        Y = sigmoid(V)
        parameters["Y" + str(i)] = Y
        parameters["V" + str(i)] = V

    return Y


def determine_error(Y_hat, Y):
    E = ((Y_hat - Y) ** 2)
    E = np.sum(E, axis=0, keepdims=True) / 2
    E = np.sum(E)

    return E


def determine_delta_rj(sample_no, layer_no, prev_delta_rj):
    global number_of_layer, train_y, parameters

    # r == L
    if layer_no == number_of_layer:
        delta_rj = np.subtract(parameters["Y" + str(layer_no)][:, sample_no], train_y[:, sample_no])
        f_dash = sigmoid_differentiated(parameters["V" + str(layer_no)][:, sample_no])
        delta_rj = np.multiply(delta_rj, f_dash)
    else:
        f_dash = sigmoid_differentiated(parameters["V" + str(layer_no)][:, sample_no])
        f_dash = f_dash.reshape((f_dash.shape[0], 1))
        delta_rj = np.dot(parameters["W" + str(layer_no + 1)], np.diagflat(f_dash))
        delta_rj = np.dot(prev_delta_rj.T, delta_rj).T

    return delta_rj


def backward_propagation():
    global number_of_layer, parameters, mu, train_x
    for i in range(1, number_of_layer + 1, 1):
        parameters["W_new" + str(i)] = copy.deepcopy(parameters["W" + str(i)])

    dataset_sz = train_x.shape[1]
    for i in range(dataset_sz):
        delta_rj = None
        for j in range(number_of_layer, 0, -1):
            # W_new = W_old - del_W * mu
            # del_W = delta_rj * y_(r-1)
            delta_rj = determine_delta_rj(i, j, delta_rj)
            delta_rj = delta_rj.reshape((delta_rj.shape[0], 1))
            sz = len(parameters["Y" + str(j - 1)][:, i])

            # W: (nodes in jth layer, nodes in j-1th layer)
            # delta_rj: (nodes in jth layer, 1)
            # Y: (1, nodes in the j-1th layer)
            parameters["W_new" + str(j)] -= mu * np.dot(delta_rj, parameters["Y" + str(j - 1)][:, i].reshape((1, sz)))

    for i in range(1, number_of_layer + 1, 1):
        parameters["W" + str(i)] = parameters["W_new" + str(i)]


def train():
    global max_itr, train_x, train_y, number_of_classes

    for i in range(max_itr):
        # Y_hat: (number of classes, dataset size)
        Y_hat = forward_propagation(train_x)
        cost = determine_error(Y_hat, train_y)
        # print("itr:", i, "cost", cost)

        backward_propagation()


def test():
    global test_x, test_y, parameters, number_of_layer, nodes_in_each_layer

    correctly_classified = 0
    misclassified = 0
    result = open("result.txt", "w")

    Y_hat = forward_propagation(test_x)
    for i in range(test_x.shape[1]):
        mx = Y_hat[0, i]
        predicted = 1
        for j in range(Y_hat.shape[0]):
            if Y_hat[j, i] > mx:
                mx = Y_hat[j, i]
                predicted = j + 1

        actual_class = test_y[i]
        if actual_class == predicted:
            correctly_classified += 1
        else:
            misclassified += 1
            result.write(
                "sample no: " + str(i + 1) + ". feature values: " + str(test_x[:, i]) + ". actual class: " + str(
                    actual_class) + ". predicted class:" + str(predicted) + "\n")

    accuracy = (correctly_classified * 100) / (misclassified + correctly_classified)

    result.write("correctly classified: " + str(correctly_classified) + "\n")
    result.write("misclassified: " + str(misclassified) + "\n")
    result.write("accuracy: " + str(accuracy))


if __name__ == "__main__":
    read_dataset()
    scale_data()

    networks = [
        [number_of_features, 3, 3, number_of_classes],
        # [number_of_features, 6, number_of_classes],
        # [number_of_features, 5, 6, 7, number_of_classes],
        # [number_of_features, 2, 4, 5, 6, number_of_classes],
        # [number_of_features, 4, 3, 5, number_of_classes],
        # [number_of_features, 3, 4, number_of_classes],
        # [number_of_features, 5, 6, 7, number_of_classes],
        # [number_of_features, 7, 5, 4, number_of_classes],
        # [number_of_features, 3, 2, 6, number_of_classes],
        # [number_of_features, 39, 22, 2, 28, 31, number_of_classes],
        # [number_of_features, 5, 6, 7, 8, 15, 8, 7, 6, 5, number_of_classes],
        # [number_of_features, 10, 15, 30, 30, number_of_classes],
        # [number_of_features, 6, 2, 4, number_of_classes],
        # [number_of_features, 5, 2, 5, number_of_classes],
        # [number_of_features, 13, 12, 16, 20, number_of_classes],
    ]

    for n in networks:
        initialize_parameters(n)
        train()
        test()
