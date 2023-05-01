import numpy as np
import matplotlib.pyplot as plt



## -- Functions related to logistic regression start here -- ##

def precidt(theta, X):
    return theta*X


def computeCost(X, y, theta, m):
    y1 = precidt(theta, X)
    y1 = np.sum(y1, axis=1)
    return sum(np.sqrt((y1-y)**2))/(2*m)


def gradientDescent(X, y, theta, m, alpha, num_iterations):
    J = []  # cost function in each iterations
    k = 0
    while k < num_iterations:
        y1 = precidt(theta, X)
        y1 = np.sum(y1, axis=1)
        for i in range(0, X.shape[1]):
            theta[0, i] = theta[0, i] - alpha*(sum((y1-y)*X[:, i])/m)
        j = computeCost(X, y, theta, m)
        J.append(j)
        k += 1
    return J, j, theta


def multi_logistic_regression(X_train, y_train, X_test, learning_rate=0.005, num_iterations=1000):
    # add a bias term to the input data
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    #normalize the data
    X_train = X_train / X_train.max(axis=0)
    X_test = X_test / X_test.max(axis=0)

    # initialize the weight vector - thetas
    theta = np.zeros([1, 12])
    # get number of samples m
    m = X_train.shape[0]

    J, j, theta = gradientDescent(
        X_train, y_train, theta, m, learning_rate, num_iterations)

    y_pred = precidt(theta, X_test)
    y_pred = np.sum(y_pred, axis=1)

    return y_pred, J


def plotCost(costs):

    # plot data
    plt.plot(costs)

    # scatter plot labels and axis ticks
    plt.title('Cost over Iterations (Learning rate at 0.005)', fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Cost', fontsize=20)
    #ax.set(xlim=(0, 100), ylim=(0, 100))

    plt.show()


def confusionMatrix(y_test, y_pred):
    cMatrix = np.zeros([11, 11])
    for i in range(y_test.shape[0]):
        cMatrix[y_test[i]][round(
            y_pred[i])] = cMatrix[y_test[i]][round(y_pred[i])] + 1

    return cMatrix

## -- Functions related to logistic regression end here -- ##
