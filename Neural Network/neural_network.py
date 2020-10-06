import numpy as np
from sklearn import preprocessing

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


def scale_data(data):
    print(data.shape)


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
    train_y = np.array(classes, dtype=float).reshape((len(lines), 1))

    assert (train_x.shape == (len(lines), number_of_features))
    assert (train_y.shape == (len(lines), 1))

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
        parameters["W" + str(i)] = np.random.rand(nodes_in_each_layer[i] + 1, nodes_in_each_layer[i - 1] + 1) * 0.01
        assert (parameters["W" + str(i)].shape == (nodes_in_each_layer[i] + 1, nodes_in_each_layer[i - 1] + 1))


def forward_propagation(input_vector):
    global number_of_layer, parameters

    Y = np.array(input_vector)
    Y = np.append(Y, 1)
    Y = Y.reshape(len(Y), 1)

    for i in range(1, number_of_layer + 1, 1):
        V = np.dot(parameters["W" + str(i)], Y)
        Y = sigmoid(V)
        Y[Y.shape[0] - 1][0] = 1

        # remove the extra row used for bias from final output
        if i == number_of_layer:
            Y = np.delete(Y, Y.shape[0] - 1, 0)


def backward_propagation():
    None


def train():
    global max_itr, train_x

    dataset_size = len(train_x)
    for i in range(max_itr):
        for j in range(dataset_size):
            forward_propagation(train_x[j])


if __name__ == "__main__":
    global train_x

    read_dataset()
    scale_data(train_x)
    initialize_parameters()
    # train()
