import numpy as np
import matplotlib.pyplot as plt


def build_design_matrix(X, degree):
    rows = [X ** p for p in range(degree, 0, -1)]
    rows.append(np.ones_like(X))
    return np.vstack(rows)


def fit_polynomial(X, Y, degree):
    X_ = build_design_matrix(X, degree)
    R = Y @ X_.T @ np.linalg.inv(X_ @ X_.T)
    return R


def predict_polynomial(R, i, degree):
    j = np.zeros_like(i, dtype=float)
    for p in range(degree + 1):
        j += R[degree - p] * i ** p
    return j


def plot_result(X, Y, i, j, line_color='b'):
    plt.plot(X, Y, '.r')
    plt.plot(i, j, line_color)
    plt.show()


if __name__ == "__main__":
    N = 50
    X = np.random.rand(N)

    # Degree 1 fit on a sine curve (underfits on purpose)
    Y = 3 * np.sin(3 * X + 0.5) + 0.5 * np.random.randn(N)
    R1 = fit_polynomial(X, Y, 1)
    i1 = np.array([0, 1])
    j1 = predict_polynomial(R1, i1, 1)
    plot_result(X, Y, i1, j1)

    # Degree 2 fit on the same sine curve
    R2 = fit_polynomial(X, Y, 2)
    i2 = np.arange(0, 1, 0.01)
    j2 = predict_polynomial(R2, i2, 2)
    plot_result(X, Y, i2, j2, line_color='g')

    # Degree 4 fit on a higher-frequency sine curve
    Y1 = 3 * np.sin(8 * X + 0.5) + 0.5 * np.random.randn(N)
    R3 = fit_polynomial(X, Y1, 4)
    i3 = np.arange(0, 1, 0.01)
    j3 = predict_polynomial(R3, i3, 4)
    plot_result(X, Y1, i3, j3)
