import random
import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *

# Load the training data and split it into features and labels


def load_data():
    data = np.loadtxt("train.dat")
    return data

def transform(X):
    transform_X = np.array([1 for _ in range(X.shape[0])])
    for i in range(X.shape[1]):
        transform_X = np.column_stack((transform_X, X[:,i]))
        for j in range(i, X.shape[1]):
            transform_X = np.column_stack((transform_X, X[:, i] * X[:, j]))
            for k in range(j, X.shape[1]):
                transform_X = np.column_stack(
                    (transform_X, X[:, i] * X[:, j] * X[:, k])
                )
    return transform_X

def random_split(data, K):
    # randomly shuffle data
    random.shuffle(data)
    # split data into folded sets
    X = data[:, :-1]
    y = data[:, -1]
    X_fold = [X[i : i + K] for i in range(0, len(X), K)]
    y_fold = [y[j : j + K] for j in range(0, len(y), K)]
    # print(X_fold[0], y_fold[0])
    return X_fold, y_fold


if __name__ == "__main__":
    data = load_data()
    K = 40
    num_experiments = 128
    log_lambdas = []
    lambdas = [1e-6, 1e-4, 1e-2, 1e0, 1e2]
    C = np.divide(1, np.multiply(2, lambdas))
    
    for i in range(num_experiments):
        data = load_data()
        # Set a different random seed for each experiment
        np.random.seed(i)
        X = data[:, :-1]
        y = data[:, -1]
        # X_fold, y_fold = random_split(data, K)
        max_acc = 0
        best_lambda = 1e-6
        X = transform(X)
        prob = problem(y, X)
        for j in range(len(lambdas)):
            ## V-fold
            param = parameter(f"-s 0 -c {C[j]} -e 0.000001 -v 5 -q")
            p_acc = train(prob, param)
            # p_labels, p_acc, p_vals = predict(valid_y, valid_X, model)

            if p_acc >= max_acc and p_acc != 0:
                if p_acc == max_acc:
                    if best_lambda < lambdas[j]:
                        # print("update")
                        best_lambda = lambdas[j]
                else:
                    # print("update")
                    max_acc = p_acc
                    best_lambda = lambdas[j]
            
        log_lambdas.append(np.log10(best_lambda))

    print(log_lambdas)
    plt.hist(log_lambdas, bins=20)
    plt.xlabel("log10(lambda)")
    plt.ylabel("Frequency")
    plt.title("Distribution of log10(lambda) over 128 experiments")
    plt.savefig("regularized_LR_12_1.png")
    # save_model('model_file', model)
