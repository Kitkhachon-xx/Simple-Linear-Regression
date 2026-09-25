import numpy as np
import matplotlib.pyplot as plt



def cal_truth(X):
    y = 3 * X + 1 + 0.5*np.random.randn(N)
    return y

def knn_regressor(data, Y, query, k):
    o = []
    for i in query:
        d = np.abs(data - i)
        d_s = np.argsort(d)[:k]
        o.append(np.mean(Y[d_s]))
    plt.plot(data, Y, '.r')
    plt.plot(query, o, 'b-')
    #plt.show()
    return o

def cal_error(data, Y, k):
    o = []
    for i in data:
        d = np.abs(data - i)
        d_s = np.argsort(d)[:k]
        o.append(np.mean(Y[d_s]))
    #print(f"Predicted values: {o}")

    mae = np.mean(np.abs(Y - o))     # MAE
    mse = np.mean((Y - o)**2)
    rmse = np.sqrt(mse)
    #print(f"Mean Absolute Error: {mae}")
    #print(f"Mean Squared Error: {mse}")
    #print(f"Root Mean Squared Error: {rmse}")
    return mae, mse, rmse

if __name__ == "__main__":
    N = 50
    d = 2
    X = np.random.rand(N)
    print(f"Input values: {X}")
    Y = cal_truth(X)
    print(f"True values: {Y}")
    i = np.array([0, 1])
    k = range(1, 11)
    for k in range(1, 11):
        print(f"K: {k}")
        knn_regressor(X, Y, i, k)
        mae, mse, rmse = cal_error(X, Y, k)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
