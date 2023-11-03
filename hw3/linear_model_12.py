import numpy as np
import matplotlib.pyplot as plt

# Function to generate data based on the given process
def generate_data(N, mean, cov, label):
    # [1, 1, 1, ...]^T + x1 + x2 stack in horizontal direction
    x = np.hstack((np.ones((N, 1)), np.random.multivariate_normal(mean, cov, N)))
    y = label * np.ones((N, 1))
    return x, y

# Generate training data
N_train = 256
mean_pos = [3, 2]
cov_pos = [[0.4, 0], [0, 0.4]]

N_outlier = 256
mean_out = [0, 6]
cov_out = [[0.1, 0], [0, 0.3]]

mean_neg = [5, 0]
cov_neg = [[0.6, 0], [0, 0.6]]

# Generate test data
N_test = 4096
X_test_pos, y_test_pos = generate_data(N_test // 2, mean_pos, cov_pos, 1)
X_test_neg, y_test_neg = generate_data(N_test // 2, mean_neg, cov_neg, -1)

X_test = np.vstack((X_test_pos, X_test_neg))
y_test = np.vstack((y_test_pos, y_test_neg))

def linear_regression(X, y):
    w_lin = np.linalg.inv(X.T @ X) @ X.T @ y
    return w_lin

def logistic_regression(X, y):
    lr = 0.1
    w_log = np.zeros((3, 1))
    for i in range(500):
        w_log = w_log + lr * 1 / N_train * (X.T @ (y * (1 / (1 + np.exp(y * (X @ w_log))))))
    return w_log

num_experiments = 128
e01_values_A = []
e01_values_B = []

for i in range(num_experiments):
    # Set a different random seed for each experiment
    np.random.seed(i)
    
    X_train_pos, y_train_pos = generate_data(N_train // 2, mean_pos, cov_pos, 1)
    X_train_out, y_train_out = generate_data(N_outlier, mean_out, cov_out, 1)
    X_train_neg, y_train_neg = generate_data(N_train // 2, mean_neg, cov_neg, -1)

    X_train = np.vstack((X_train_pos, X_train_neg))
    X_train = np.vstack((X_train, X_train_out))
    y_train = np.vstack((y_train_pos, y_train_neg))
    
    # Shuffle the training data
    indices = np.random.permutation(N_train)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # # Run linear regression
    w_lin = linear_regression(X_train, y_train)

    # Calculate 0/1 in-sample error
    # print(((np.array([1,2]) * np.array([1,-1]))))
    count = 0
    for i in range(len(y_test)):
        if np.sign(X_test[i] @ w_lin) * y_test[i] != 1:
            count += 1
            # print(X_test[i], y_test[i], np.sign(X_test[i] @ w_lin))
        
    e01_A = count/len(y_test)
    e01_values_A.append(e01_A)

    
    # Run logistic regression
    w_log = logistic_regression(X_train, y_train)
    
    count = 0
    for i in range(len(y_test)):
        if np.sign(X_test[i] @ w_log) * y_test[i] != 1:
            count += 1
    # Calculate zero-one loss for logistic regression
    e01_B = count/len(y_test)
    e01_values_B.append(e01_B)

# Plot scatter plot for [E0/1(A(D)), E0/1(B(D))]
plt.scatter(e01_values_A, e01_values_B, alpha=0.5)
plt.xlabel('E0/1(A(D))')
plt.ylabel('E0/1(B(D))')
plt.title('Comparison of Linear Regression (A) and Logistic Regression (B)')
plt.savefig('linear_model_12.png')

# Calculate median E0/1 values
median_e01_A = np.median(e01_values_A)
median_e01_B = np.median(e01_values_B)

print(f"Median E0/1(A(D)) over 128 experiments: {median_e01_A}")
print(f"Median E0/1(B(D)) over 128 experiments: {median_e01_B}")
