import numpy as np

seed_value = 107
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


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def scale_data():
    global train_x

    dataset_size = train_x.shape[0]
    for i in range(dataset_size):
        mean = np.mean(train_x[i])
        sd = np.mean(train_x[i])

        train_x[i] = (train_x[i] - mean) / sd


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

    train_x = np.array(features, dtype=float)
    train_y = np.zeros((number_of_classes, len(lines)))
    for i in range(len(lines)):
        train_y[classes[i] - 1][i] = 1

    assert (train_x.shape == (len(lines), number_of_features))
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

    test_x = np.array(features, dtype=float)
    test_y = np.array(classes, dtype=float).reshape((len(lines), 1))

    assert (test_x.shape == (len(lines), number_of_features))
    assert (test_y.shape == (len(lines), 1))

    # free memories
    del features, classes, lines


def initialize_parameters():
    global number_of_layer, nodes_in_each_layer, number_of_classes, number_of_features, \
        parameters

    # input layer -> hidden layers -> output layers
    number_of_layer = 3
    nodes_in_each_layer = [number_of_features, 3, 3, number_of_classes]

    assert (number_of_layer == len(nodes_in_each_layer) - 1)

    for i in range(1, number_of_layer + 1, 1):
        parameters["W" + str(i)] = np.random.rand(nodes_in_each_layer[i], nodes_in_each_layer[i - 1] + 1) * 0.01
        assert (parameters["W" + str(i)].shape == (nodes_in_each_layer[i], nodes_in_each_layer[i - 1] + 1))


def forward_propagation(input_vector):
    global number_of_layer, parameters

    Y = np.array(input_vector)
    Y = np.append(Y, 1)
    Y = Y.reshape(len(Y), 1)

    for i in range(1, number_of_layer + 1, 1):
        V = np.dot(parameters["W" + str(i)], Y)
        Y = sigmoid(V)

        # add an extra row for the bias
        if i < number_of_layer:
            Y = np.append(Y, 1)

        Y = Y.reshape(len(Y), 1)

    # print("output",Y)
    return Y


def determine_error(Y_hat, Y):
    E = np.sum((Y_hat - Y) ** 2) / 2
    return E


def backward_propagation():
    None


def train():
    global max_itr, train_x, train_y, number_of_classes

    dataset_size = train_x.shape[0]
    for i in range(max_itr):
        for j in range(dataset_size):
            Y = forward_propagation(train_x[j])
            E = determine_error(Y, train_y[:, j].reshape(number_of_classes, 1))


if __name__ == "__main__":
    read_dataset()
    scale_data()
    initialize_parameters()
    train()
