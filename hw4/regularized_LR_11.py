import random
import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *

# Load the training data and split it into features and labels

def load_data():
    data = np.loadtxt("./train.dat")
    return data

def transform(X):
    transform_X = np.array([1 for _ in range(X.shape[0])])
    for i in range(X.shape[1]):
        transform_X = np.column_stack((transform_X, X[:, i]))
        for j in range(i, X.shape[1]):
            transform_X = np.column_stack((transform_X, X[:, i] * X[:, j]))
            for k in range(j, X.shape[1]):
                transform_X = np.column_stack(
                    (transform_X, X[:, i] * X[:, j] * X[:, k])
                )
    return transform_X

def random_split(data):
    # randomly shuffle data
    # np.random.seed(42)
    random.shuffle(data)
    # split data into training and validation sets
    train_data = data[:120]
    valid_data = data[120:]
    # print(train_data.shape, valid_data.shape)
    train_X = train_data[:, :-1]
    train_y = train_data[:, -1]
    # print(train_X.shape, train_y.shape)
    valid_X = valid_data[:, :-1]
    valid_y = valid_data[:, -1]
    # print(valid_X.shape, valid_y.shape)
    return train_X, train_y, valid_X, valid_y


if __name__ == "__main__":
    data = load_data()
    # print(data.shape)

    num_experiments = 128
    log_lambdas = []
    for i in range(num_experiments):
        # Set a different random seed for each experiment
        # np.random.seed(i)
        data = load_data()
        train_X, train_y, valid_X, valid_y = random_split(data)
        train_X = transform(train_X)
        valid_X = transform(valid_X)
        prob = problem(train_y, train_X)
        
        lambdas = [1e-6, 1e-4, 1e-2, 1e0, 1e2]
        C = np.divide(1, np.multiply(2, lambdas))
        best_lambda = 1e-6
        acc = 0
        for i in range(len(lambdas)):
            param = parameter(f"-s 0 -c {C[i]} -e 0.000001 -q")
            model = train(prob, param)
            p_labels, p_acc, p_vals = predict(valid_y, valid_X, model)
            if p_acc[0] >= acc and p_acc[0] != 0:
                if p_acc[0] == acc:
                    if best_lambda < lambdas[i]:
                        print("update")
                        acc = p_acc[0]
                        best_lambda = lambdas[i]
                        best_model = model
                else:
                    # print(acc)
                    acc = p_acc[0]
                    best_lambda = lambdas[i]
                    best_model = model
        
        log_lambdas.append(np.log10(best_lambda))

    print(log_lambdas)
    plt.hist(log_lambdas, bins=20)
    plt.xlabel("log10(lambda)")
    plt.ylabel("Frequency")
    plt.title("Distribution of log10(lambda) over 128 experiments")
    plt.savefig("regularized_LR_11.png")
    # save_model('model_file', model)
