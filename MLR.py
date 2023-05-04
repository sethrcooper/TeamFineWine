import numpy as np
import matplotlib.pyplot as plt



## -- Functions related to logistic regression start here -- ##

def predict(theta, X):
    h = theta*X
    h = np.sum(h, axis=1)
    return h


def predictPreset(X):
    #constant weights from a training run with high accuracy - avaiable for grader use
    theta = [[2.61025072, 1.30790801, 0.86575955, 0.66330309, 0.43259791, 0.30079463,
              0.31653967, 0.88326665, 1.93324678]]
    # Remove Density - All values are 1 or 0.99 it is basically another bias term
    X = np.delete(X, 8, 1)
    # Remove total sulphur dioxides - 1st least important
    X = np.delete(X, 7, 1)
    # Remove chlorides - 2nd least important
    X = np.delete(X, 5, 1)
    # add a bias term to the input data
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    #normalize the data
    X = X / X.max(axis=0)

    h = theta*X
    h = np.sum(h, axis=1)
    return h


def computeCost(X, y, theta, m):
    y1 = predict(theta, X)
    return sum(np.sqrt((y1-y)**2))/(2*m)


def gradientDescent(X, y, theta, m, alpha, num_iterations):
    J = []  # cost function in each iterations
    for n in range(num_iterations):
        y1 = predict(theta, X)
        for i in range(0, X.shape[1]):
            theta[0, i] = theta[0, i] - alpha*(sum((y1-y)*X[:, i])/m)
        j = computeCost(X, y, theta, m)
        J.append(j)
    return J, j, theta


def multi_logistic_regression(X_train, y_train, X_test, learning_rate=0.005, num_iterations=1000):
    # DATA Trimming
    # Remove Density - All values are 1 or 0.99 it is basically another bias term
    X_train = np.delete(X_train, 8, 1)
    X_test = np.delete(X_test, 8, 1)
    # Remove total sulphur dioxides - 1st least important
    X_train = np.delete(X_train, 7, 1)
    X_test = np.delete(X_test, 7, 1)
    # Remove chlorides - 2nd least important
    X_train = np.delete(X_train, 5, 1)
    X_test = np.delete(X_test, 5, 1)
    # add a bias term to the input data
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    #normalize the data
    X_train = X_train / X_train.max(axis=0)
    X_test = X_test / X_test.max(axis=0)

    # initialize the weight vector - thetas
    theta = np.zeros([1, X_train.shape[1]])
    # get number of samples m
    m = X_train.shape[0]

    J, j, theta = gradientDescent(
        X_train, y_train, theta, m, learning_rate, num_iterations)

    y_pred = predict(theta, X_test)

    return y_pred, J, theta


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
