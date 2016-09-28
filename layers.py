from abc import ABCMeta, abstractmethod
import numpy as np


class Layer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fprop(self, parentOutput): raise NotImplementedError()

    @abstractmethod
    def bprop(self, childGradient): raise NotImplementedError()


class Output(object):
    """
    Soft-max output with cross-entropy entropy loss function
    """
    def __init__(self):
        pass

    def bprop(self, y, fx):
        e = np.zeros(len(fx))
        e[y] = 1
        myGradient = -(e-fx)
        return myGradient

    def fprop(self, parentOutput):
        softmax = np.exp(parentOutput)/sum(np.exp(parentOutput))
        return softmax

    def fprop_loss(self, parentOutput, label):
        softmax = self.fprop(parentOutput)
        loss = -np.log(softmax[label.astype(int), range(0, len(label))])
        return loss


class PreActivation(object):
    def __init__(self, weights, biases):
        """
        Linear pre-activation layer
        :param parent: parent Layer object
        :param child: child Layer object
        :param W: Weights object
        :param b: Biases object
        """
        self.weights = weights
        self.biases = biases

    def fprop(self, parentOutput, W, b):
        output = (np.dot(W.T, parentOutput).T + b).T
        return output

    def bprop(self, childGradient, myLastOutput):
        myGradient = np.multiply(childGradient, np.multiply(sigmoid(myLastOutput), (1-sigmoid(myLastOutput))))
        return myGradient


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


class Activation(object):
    def __init__(self):
        """
        Sigmoid activation layer
        :param parent: parent Layer object
        :param child: child Layer object
        """

    def fprop(self, parentOutput):
        output = sigmoid(parentOutput)
        return output

    def bprop(self, childGradient, childWeights):
        myGradient = np.dot(childWeights, childGradient)
        return myGradient


class Weights(object):
    def __init__(self, W0):
        """
        Weight vectors
        :param W0: Initialized values, numpy array (ninputs x noutputs)
        """
        self._W = W0

    def bprop(self, childGradient, lastHidden):
        myGradient = np.outer(childGradient, lastHidden)
        return myGradient

    def fprop(self):
        return self._W

    def update(self, step):
        self._W += step


class Biases(object):
    def __init__(self, b0):
        """
        Vector of biases
        :param b0: Initialized values, numpy array (noutputs)
        """
        self._b = b0

    def bprop(self, childGradient):
        return childGradient

    def fprop(self):
        return self._b

    def update(self, step):
        self._b += step