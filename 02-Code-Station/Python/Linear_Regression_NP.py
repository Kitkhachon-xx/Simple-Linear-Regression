import numpy as np
import matplotlib.pyplot as plt


def generate_data(N):
    X = np.random.rand(N)
    Y = 3 * X + 1 + 0.5 * np.random.randn(N)
    return X, Y


def fit_linear(X, Y):
    # R = Y * X.T * (X * X.T)^-1
    X_ = np.vstack((X, np.ones(len(X))))
    R = Y @ X_.T @ np.linalg.inv(X_ @ X_.T)
    return R


def predict_linear(R, i):
    return R[0] * i + R[1]


def plot_result(X, Y, i, j):
    plt.plot(X, Y, '.r')
    plt.plot(i, j, 'b')
    plt.show()


if __name__ == "__main__":
    N = 50
    X, Y = generate_data(N)

    R = fit_linear(X, Y)
    print(R)

    i = np.array([0, 1])
    j = predict_linear(R, i)
    plot_result(X, Y, i, j)
