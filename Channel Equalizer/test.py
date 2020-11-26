import numpy as np
from collections import defaultdict
import scipy.stats
from math import log

def bin2int(arr):
    y = 0
    for i, j in enumerate(arr):
        y += j<<i
    return y

# np.random.seed(11111)
smallnumber = 0.0000000000000000000001 # so that logarithm doesn't break

with open("parameters.txt", "r") as config_file:
    n, l = map(int, config_file.readline().split())
    weights = list(map(float, config_file.readline().split()))
    sigma_squared = float(config_file.readline()) + smallnumber
    sigma = np.sqrt(sigma_squared)

with open("test/train.txt") as train_file:
    train_bits = list(map(int, train_file.readline()))

samples = []
for i in range(len(train_bits)-n+1):
    # print(train_bits[i:i+n])
    samples.append(train_bits[i:i+n])

samples = np.array(samples)
print(samples.shape)
weights = np.array(weights)
I = samples.dot(weights)
for i in range(len(I)):
    I[i] += np.random.normal(loc=0.0, scale=sigma) # adding noise
# print(X)

prior_probabilities = defaultdict(float)
transition_probabilities = np.zeros((2**n, 2**n))
clusters = defaultdict(list)
cluster_means = defaultdict(float)

clusters[bin2int(samples[0])].append(I[0])
prior_probabilities[bin2int(samples[0])] += 1
for i in range(1, len(I)):
    prior_probabilities[bin2int(samples[i])] += 1
    transition_probabilities[bin2int(samples[i])][bin2int(samples[i - 1])] += 1
    clusters[bin2int(samples[i])].append(I[i])

print("\n\n\nTransition count:")
print(transition_probabilities)

transition_probabilities /= transition_probabilities.sum(axis=1, keepdims=True)

for i in range(2**n):
    cluster_means[i] = (np.average(clusters[i]))

for i in range(2**n):
    prior_probabilities[i] /= len(I)

######################################################################################################
print("\n\n\nPrior probabilities:")
for i in range(2**n):
    print(i, "\t :\t",  round(prior_probabilities[i], 2))
# print(prior_probabilities)

print("\n\n\nTransition probabilities:")
for i in range(2**n):
    print(i, "\t : ", end="")
    for j in range(2**n):
        print("\t%.2f" % (transition_probabilities[i][j]), end="")
    print("")
# print(transition_probabilities)

# print(clusters)

print("\n\n\nObservation means:")
for i in range(2**n):
    print(i, "\t : \t%.2f" %(cluster_means[i]))
# print(cluster_means)


with open("test/test.txt") as test_file:
    test_bits = list(map(int, test_file.readline()))

res = []
test_samples = []
for i in range(len(test_bits)-n+1):
    # print(train_bits[i:i+n])
    test_samples.append(test_bits[i:i+n])

test_samples = np.array(test_samples)
# print(test_samples)
X = test_samples.dot(weights)
for i in range(len(X)):
    X[i] += np.random.normal(loc=0.0, scale=sigma) # adding noise
# print(X)

cost_and_parent = []
x = X[0]
temp_dict = {}
for j in range(2**n):
    probability_x_given_cluster_j = scipy.stats.norm.pdf(x, cluster_means[j], sigma)
    ln_p_x_given_w = log(probability_x_given_cluster_j + smallnumber)
    probability_w_one = prior_probabilities[j]
    ln_p_w_one = log(probability_w_one + smallnumber)
    cost = ln_p_w_one + ln_p_x_given_w
    parent = None
    temp_dict[j] = (cost, parent)
cost_and_parent.append(temp_dict)

for i in range(1, len(X)):
    x = X[i]
    temp_dict = {}
    for j in range(2 ** n):
        probability_x_given_cluster_j = scipy.stats.norm.pdf(x, cluster_means[j], sigma)
        ln_p_x_given_w = log(probability_x_given_cluster_j + smallnumber)
        max_cost = -float("inf")
        parent = None
        for k in range(2**n):
            probability_wj_given_wk = transition_probabilities[j][k]
            ln_p_wj_given_wk = log(probability_wj_given_wk + smallnumber) #adding smallnumber to prevent log(zero)
            cost = cost_and_parent[i-1][k][0] + ln_p_x_given_w + ln_p_wj_given_wk
            if cost > max_cost:
                max_cost = cost
                parent = k
                temp_dict[j] = (max_cost, parent)
    cost_and_parent.append(temp_dict)

last_dict = cost_and_parent[-1]
last_class = None
parent = None
max_cost = -float("inf")
for j in range(2**n):
    cost = last_dict[j][0]
    if cost > max_cost:
        max_cost = cost
        last_class = j
        parent = last_dict[j][1]

decision = []
for i in range(len(cost_and_parent)-1, 0, -1):
    dictionary = cost_and_parent[i]
    parent = dictionary[last_class][1]
    # print(parent)
    decision.append(last_class)
    last_class = parent

last_dict = cost_and_parent[0]
last_class = None
max_cost = -float("inf")
for j in range(2**n):
    cost = last_dict[j][0]
    if cost > max_cost:
        max_cost = cost
        last_class = j
decision.append(last_class)

res = []
for i in list(reversed(decision)):
    res.append(np.binary_repr(i, width=n)[0])

print("\n\n\nOriginal bits\t : ", test_bits[n-1:])
predictions = list(map(int, res))
print("Predictions\t : ", predictions)

res = np.array(predictions).reshape(1, len(predictions))
test = np.array(test_bits[n-1:]).reshape(1, len(test_bits)-n+1)

diff = res-test
print("\n\nAccuracy : ",  100 - len(res[res!=test]) / len(predictions) * 100, "%")