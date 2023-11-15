import numpy as np
from liblinear.liblinearutil import *

# Load the training data and split it into features and labels


def load_data():
    data = np.loadtxt("train.dat")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def transform(X):
    transform_X = np.array([1 for _ in range(X.shape[0])])
    print(transform_X)
    for i in range(X.shape[1]):
        transform_X = np.column_stack((transform_X, X[:, i]))
        for j in range(i, X.shape[1]):
            transform_X = np.column_stack((transform_X, X[:, i] * X[:, j]))
            for k in range(j, X.shape[1]):
                transform_X = np.column_stack(
                    (transform_X, X[:, i] * X[:, j] * X[:, k])
                )
    return transform_X

if __name__ == "__main__":
    X, y = load_data()
    lambdas = [1e-6, 1e-4, 1e-2, 1e0, 1e2]
    C = np.divide(1, np.multiply(2, lambdas))

    transform_X = transform(X)

    # print(transform_X.shape)
    acc = 0
    for i in range(len(lambdas)):
        param = parameter(f"-s 0 -c {C[i]} -e 0.000001 -q")
        prob = problem(y, transform_X)
        model = train(prob, param)
        p_labels, p_acc, p_vals = predict(y, transform_X, model)
        print(p_acc)
        if p_acc[0] > acc:
            acc = p_acc[0]
            best_lambda = lambdas[i]
            best_model = model

    print(f"best_lambda = {best_lambda}")
    print(f"best_acc = {acc}")
