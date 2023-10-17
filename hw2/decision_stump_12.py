import random
import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(size):
    x = sorted([random.uniform(-1, 1) for _ in range(size)])
    y = [np.sign(np.sign(xi)+(2.5 * random.random() - 1.25)) for xi in x]
    # flip_num = random
    # flip_indices = random.sample(range(size), 6)
    # for i in flip_indices: 
    #     y[i] = -y[i]
    return x, y

def hypothesis(x, theta, s):
    return s * np.sign(x - theta)

def calculate_ein(dataset, s, theta):
    x, y = dataset
    error = 0
    for i in range(len(y)):
        if hypothesis(x[i], theta, s) != y[i]:
            error += 1
    return error / len(x)

def decision_stump(dataset):
    x, y = dataset
    x_prime = [(x[i] + x[i+1]) / 2 for i in range(len(x)-1)]
    thetas = [-1] + x_prime
    hypotheses = [(-1, theta) for theta in thetas] + [(1, theta) for theta in thetas]
    # print(hypotheses)
    best_ein = float('inf')
    g = (random.choice((-1, 1)), random.uniform(-1, 1))
    print(g)
    # for s, theta in hypotheses:
    #     ein = calculate_ein(dataset, s,theta)
    #     if ein <= best_ein:
    #         if ein == best_ein:
    #             if s * theta < g[0] * g[1]:
    #                 g = (s, theta)
    #         else:
    #             best_ein = ein
    #             g = (s, theta)
            # print(best_ein, g)
    return g

def calculate_eout(dataset, s, theta):
    x, y = dataset
    error = 0.5 - 0.4 * s + 0.4 * s * abs(theta)
    return error

ein_list = []
eout_list = []
for i in range(2000):
    dataset = generate_dataset(size = 8)
    g = decision_stump(dataset)
    ein = calculate_ein(dataset, g[0], g[1])
    ein_list.append(ein)
    eout = calculate_eout(dataset, g[0], g[1])
    eout_list.append(eout)

plt.scatter(ein_list, eout_list)
plt.xlabel('Ein')
plt.ylabel('Eout')
# plt.show()
plt.savefig('decision_stump_12.png')

median = np.median(np.array(eout_list) - np.array(ein_list))
print('Median of Eout(g) - Ein(g):', median)
