import numpy as np

seed_value = 1
np.random.seed(seed_value)

test_x = None
test_y = None
train_x = None
train_y = None

number_of_features = None
number_of_classes = None

test_file_name = "./dataset/testNN.txt"
train_file_name = "./dataset/trainNN.txt"

number_of_layer = None
nodes_in_each_layer = []

parameters = {}
max_itr = 1
mu = 0.01


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def sigmoid_differentiated(val):
    None


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

    assert (train_x.shape == (number_of_features, len(lines)))
    assert (train_y.shape == (number_of_classes, len(lines)))

    # read dataset to test
    test_file = open(test_file_name)
    lines = test_file.readlines()

    features = []
    classes = []
    for line in lines:
        values = line.strip().split()
        features.append(values[:number_of_features])
        classes.append(int(values[number_of_features]))

    test_x = np.array(features, dtype=float).T
    test_y = np.array(classes, dtype=float).reshape((len(lines), 1))

    assert (test_x.shape == (number_of_features, len(lines)))
    assert (test_y.shape == (len(lines), 1))

    # free memories
    del features, classes, lines


def initialize_parameters():
    global number_of_layer, nodes_in_each_layer, number_of_classes, number_of_features, \
        parameters

    # input layer -> hidden layers -> output layers
    number_of_layer = 4
    nodes_in_each_layer = [number_of_features, 3, 3, 3, number_of_classes]

    assert (number_of_layer == len(nodes_in_each_layer) - 1)

    for i in range(1, number_of_layer + 1, 1):
        parameters["W" + str(i)] = np.random.randn(nodes_in_each_layer[i], nodes_in_each_layer[i - 1])
        assert (parameters["W" + str(i)].shape == (nodes_in_each_layer[i], nodes_in_each_layer[i - 1]))


def forward_propagation(input_vector):
    global number_of_layer, parameters
    # 5*500 3*5
    Y = np.array(input_vector)
    for i in range(1, number_of_layer + 1, 1):
        V = np.dot(parameters["W" + str(i)], Y)
        Y = sigmoid(V)

    return Y


def determine_error(Y_hat, Y):
    E = ((Y_hat - Y) ** 2)
    E = np.sum(E, axis=0, keepdims=True) / 2
    E = np.sum(E)

    return E


def backward_propagation():
    None


def train():
    global max_itr, train_x, train_y, number_of_classes

    for i in range(max_itr):
        Y_hat = forward_propagation(train_x)
        cost = determine_error(Y_hat, train_y)
        print(cost)

        backward_propagation()


if __name__ == "__main__":
    read_dataset()
    scale_data()
    initialize_parameters()
    train()

    # x = np.array([[3, 5], [4, 7]])
    # y = np.array([[6, 2], [5, 9]])
    #
    # ans = np.sum(x, axis=0)
    # print(ans)
