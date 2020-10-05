import numpy as np

seed_value = 107
np.random.seed(seed_value)

test_x = []
test_y = []
train_x = []
train_y = []

number_of_features = None
number_of_classes = None

test_file_name = "./dataset/testNN.txt"
train_file_name = "./dataset/trainNN.txt"

number_of_layer = None
nodes_in_each_layer = []

parameters = {}
max_itr = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_dataset():
    global number_of_features, number_of_classes, train_x, train_y, test_x, test_y, \
        train_file_name, test_file_name

    # read dataset to train
    train_file = open(train_file_name)
    lines = train_file.readlines()

    assert (len(lines) > 0)

    number_of_features = len(lines[0].strip().split()) - 1
    number_of_classes = 0

    for line in lines:
        values = line.strip().split()
        train_x.append(np.array(values[:number_of_features], dtype=float))
        train_y.append(int(values[number_of_features]))

        number_of_classes = max(number_of_classes, int(values[number_of_features]))

    # read dataset to test
    test_file = open(test_file_name)
    lines = test_file.readlines()
    for line in lines:
        values = line.strip().split()
        test_x.append(np.array(values[:number_of_features], dtype=float))
        test_y.append(int(values[number_of_features]))


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
    read_dataset()
    initialize_parameters()
    train()
