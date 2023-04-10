"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """

    labels = np.unique(train_labels)

    d, n = train_data.shape
    num_classes = labels.size

    model = dict()
    model["cond_means"] = np.empty((d, num_classes))
    model["cond_std"] = np.empty((d, num_classes))
    model["prior_probs"] = np.empty((num_classes, 1))
    for i in range(num_classes): 
        class_data = train_data[:,train_labels == i]
        model["cond_means"][:,i] = np.mean(class_data, axis=1)
        model["cond_std"][:,i] = np.std(class_data, axis=1)
        model["prior_probs"][i] = class_data.shape[1] / n

    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    cond_probs = np.empty((model["cond_means"].shape[1], data.shape[1]))
    for i in range(data.shape[1]): 
        point = (data[:,i][:,np.newaxis] - model["cond_means"]) / model["cond_std"]
        probs = np.exp(-(point * point) / 2) / np.sqrt(2*np.pi)
        cond_probs[:,i] = np.product(probs, axis=0)

    class_prob = model["prior_probs"] * cond_probs

    return np.argmax(class_prob, axis=0)