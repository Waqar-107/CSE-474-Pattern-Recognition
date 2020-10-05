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


if __name__ == "__main__":
    read_dataset()
