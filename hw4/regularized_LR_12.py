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
        transform_X = np.column_stack((transform_X, X[:, i]))
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
    print(X[K : K + K].shape, y[K : K + K].shape)
    X_fold = [X[i : i + K] for i in range(0, len(X), K)]
    y_fold = [y[j : j + K] for j in range(0, len(y), K)]
    # print(X_fold[0], y_fold[0])
    return X_fold, y_fold


if __name__ == "__main__":
    data = load_data()
    K = 40
    num_experiments = 5
    log_lambdas = []
    lambdas = [1e-6, 1e-4, 1e-2, 1e0, 1e2]
    C = np.divide(1, np.multiply(2, lambdas))

    for i in range(num_experiments):
        data = load_data()
        # Set a different random seed for each experiment
        np.random.seed(i)
        X_fold, y_fold = random_split(data, K)
        max_acc = 0
        best_lambda = 1e-6
        for j in range(len(lambdas)):
            accs = []
            ## V-fold
            for v in range(len(data) // K):
                if v == 0:
                    train_X = np.vstack(X_fold[1:])
                    train_y = np.hstack(y_fold[1:])
                    valid_X = np.array(X_fold[0])
                    valid_y = np.array(y_fold[0])
                elif v == len(data) // K - 1:
                    train_X = np.vstack(X_fold[:-1])
                    train_y = np.hstack(y_fold[:-1])
                    valid_X = np.array(X_fold[-1])
                    valid_y = np.array(y_fold[-1])
                else:
                    print(
                        v,
                        np.vstack(X_fold[:v]).shape,
                        np.vstack(X_fold[(v + 1) :]).shape,
                        np.array(y_fold[:v]).shape,
                        np.array(y_fold[(v + 1) :]).shape,
                    )
                    train_X = np.vstack(
                        (np.vstack(X_fold[:v]), np.vstack(X_fold[(v + 1) :]))
                    )
                    train_y = np.hstack(
                        (np.hstack(y_fold[:v]), np.hstack(y_fold[(v + 1) :]))
                    )
                    valid_X = np.array(X_fold[v])
                    valid_y = np.array(y_fold[v])

                    print(train_X.shape, train_y.shape)
                train_X = transform(train_X)
                valid_X = transform(valid_X)
                print(train_X)
                prob = problem(train_y, train_X)
                param = parameter(f"-s 0 -c {C[j]} -e 0.000001 -q")
                model = train(prob, param)
                p_labels, p_acc, p_vals = predict(valid_y, valid_X, model)
                accs.append(p_acc[0])
            print(np.mean(accs), max_acc, lambdas[j])
            if np.mean(accs) >= max_acc and np.mean(accs) != 0:
                if np.mean(accs) == max_acc:
                    if best_lambda < lambdas[j]:
                        print("update")
                        best_lambda = lambdas[j]
                else:
                    print("update", accs)
                    max_acc = np.mean(accs)
                    best_lambda = lambdas[j]

        log_lambdas.append(np.log10(best_lambda))

    print(log_lambdas)
    plt.hist(log_lambdas, bins=20)
    plt.xlabel("log10(lambda)")
    plt.ylabel("Frequency")
    plt.title("Distribution of log10(lambda) over 128 experiments")
    plt.savefig("regularized_LR_12.png")
