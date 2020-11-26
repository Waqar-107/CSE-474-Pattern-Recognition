import numpy as np


def to_bin(num, b):
    arr = []
    while num:
        arr.append(num % 2)
        num //= 2

    while len(arr) < b:
        arr.append(0)

    return arr[::-1]


def to_dec(num):
    x = 0
    for j in range(len(num)):
        x = x << 1 | num[j]

    return x


# read the parameters
params = open("parameters.txt", "r")
n, l = map(int, params.readline().split())
h = np.array(list(map(float, params.readline().split())))
sigma = np.sqrt(float(params.readline()))

# read the train file
train_file = open("train.txt", "r")
train_bits = train_file.readline()

# build the sample
training_samples = []
bits = list(map(int, train_bits[0: n]))
training_samples.append(bits)
for i in range(1, len(train_bits) - n + 1):
    bits.pop(0)
    bits.append(int(train_bits[i]))
    training_samples.append(bits)

training_samples = np.array(training_samples)

# determine Xk = sum(h * I) + noise
Xk = np.dot(training_samples, h) + np.random.normal(loc=0, scale=sigma)

cluster_quantity = np.power(2, n + l - 1)
transitional_probabilities = np.zeros((cluster_quantity, cluster_quantity))
for i in range(cluster_quantity):
    j = (i >> 1)
    transitional_probabilities[i][j] = 0.5

    j |= (1 << (n + l - 2))
    transitional_probabilities[i][j] = 0.5

