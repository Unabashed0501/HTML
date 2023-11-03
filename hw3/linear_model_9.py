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

mean_neg = [5, 0]
cov_neg = [[0.6, 0], [0, 0.6]]

# Generate test data
N_test = 4096
X_test_pos, y_test_pos = generate_data(N_test // 2, mean_pos, cov_pos, 1)
X_test_neg, y_test_neg = generate_data(N_test // 2, mean_neg, cov_neg, -1)

X_test = np.vstack((X_test_pos, X_test_neg))
y_test = np.vstack((y_test_pos, y_test_neg))

# Linear regression algorithm
def linear_regression(X, y):
    w_lin = np.linalg.inv(X.T @ X) @ X.T @ y
    return w_lin

num_experiments = 128
esqr_values = []

for i in range(num_experiments):
    # Set a different random seed for each experiment
    np.random.seed(i)
    
    X_train_pos, y_train_pos = generate_data(N_train // 2, mean_pos, cov_pos, 1)
    X_train_neg, y_train_neg = generate_data(N_train // 2, mean_neg, cov_neg, -1)

    X_train = np.vstack((X_train_pos, X_train_neg))
    y_train = np.vstack((y_train_pos, y_train_neg))
    
    # Shuffle the training data
    indices = np.random.permutation(N_train)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # # Run linear regression
    w_lin = linear_regression(X_train, y_train)

    # Calculate squared in-sample error
    esqr = np.mean((X_test @ w_lin - y_test)**2)
    esqr_values.append(esqr)

# Plot histogram of Esqr(wlin)
plt.hist(esqr_values, bins=50)
plt.xlabel('Esqr(wlin)')
plt.ylabel('Frequency')
plt.title('Distribution of Esqr(wlin) over 128 experiments')
plt.savefig('linear_model_9.png')

# Calculate median Esqr
median_esqr = np.median(esqr_values)
print(f"Median Esqr over 128 experiments: {median_esqr}")
