import numpy as np
from layers import Output, PreActivation, Activation, Weights, Biases
from itertools import izip
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
import cProfile
from scipy.cluster.vq import whiten


def main():
    train = np.loadtxt('digitstrain.txt', delimiter=',')
    np.random.shuffle(train)
    train_imgs = train[:, 0:-1]
    train_imgs = train_imgs - np.mean(train_imgs, axis=0)
    train_imgs = whiten(train_imgs)
    train_labels = train[:, -1]
    valid = np.loadtxt('digitsvalid.txt', delimiter=',')
    valid_imgs = valid[:, 0:-1]
    valid_imgs = valid_imgs - np.mean(valid_imgs, axis=0)
    valid_imgs = whiten(valid_imgs)
    valid_labels = valid[:, -1]

    H = [784, 100, 10]
    epochs = 5
    eta = -0.1
    net = SingleLayerNet(H)
    train_loss, valid_loss, train_error, valid_error, W = net.train(train_imgs, train_labels, valid_imgs, valid_labels,
                                                                    epochs, eta)
    plt.figure(1)
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xlabel('epoch')
    plt.ylabel(['Average Cross-Entropy Loss, epsilon=', str(eta)])
    plt.legend(['Train', 'Validation'])
    plt.show()

    plt.figure(2)
    plt.plot(train_error)
    plt.plot(valid_error)
    plt.xlabel('epoch')
    plt.ylabel('Average Classification Error')
    plt.legend(['Train', 'Validation'])
    plt.show()


class SingleLayerNet(object):
    def __init__(self, H):
        """

        :param H: List of the number of units in each layer (len = 3)
        """
        self._weights0 = Weights(np.random.uniform(low=-np.sqrt(6. / H[0]), high=np.sqrt(6. / H[0]), size=(784, H[0])))
        self._biases0 = Biases(np.zeros(H[0]))
        self._a0 = PreActivation(self._weights0, self._biases0)
        self._h0 = Activation()

        self._weights1 = Weights(
            np.random.uniform(low=-np.sqrt(6. / (H[0] + H[1])), high=np.sqrt(6. / (H[0] + H[1])), size=(H[0], H[1])))
        self._biases1 = Biases(np.zeros(H[1]))
        self._a1 = PreActivation(self._weights1, self._biases1)
        self._h1 = Activation()

        self._weights2 = Weights(
            np.random.uniform(low=-np.sqrt(6. / (H[1] + H[2])), high=np.sqrt(6. / (H[1] + H[2])), size=(H[1], H[2])))
        self._biases2 = Biases(np.zeros(H[2]))
        self._a2 = PreActivation(self._weights2, self._biases2)
        self._f = Output()

    def train(self, train_imgs, train_labels, valid_imgs, valid_labels, epochs, eta):
        """
        :param train_imgs:
        :param valid_imgs:
        :return:
        """
        train_losses = []
        valid_losses = []
        train_errors = []
        valid_errors = []
        for i in range(0, epochs):
            ## Evaluation current network
            y_hat = []
            train_losses_this_epoch = []

            #for img, label in izip(train_imgs, train_labels):
                # Forward prop
            a0_output = self._a0.fprop(train_imgs.T, self._weights0.fprop(), self._biases0.fprop())
            h0_output = self._h0.fprop(a0_output)
            a1_output = self._a1.fprop(h0_output, self._weights1.fprop(), self._biases1.fprop())
            h1_output = self._h1.fprop(a1_output)
            a2_output = self._a2.fprop(h1_output, self._weights2.fprop(), self._biases2.fprop())
            f_output = self._f.fprop(a2_output)
            train_losses_this_epoch = self._f.fprop_loss(a2_output, train_labels)
            y_hat = np.argmax(f_output, axis=0)

            misclassification_error = hamming(train_labels, y_hat)
            train_losses.append(np.mean(train_losses_this_epoch))
            train_errors.append(misclassification_error)

            a0_output = self._a0.fprop(valid_imgs.T, self._weights0.fprop(), self._biases0.fprop())
            h0_output = self._h0.fprop(a0_output)
            a1_output = self._a1.fprop(h0_output, self._weights1.fprop(), self._biases1.fprop())
            h1_output = self._h1.fprop(a1_output)
            a2_output = self._a2.fprop(h1_output, self._weights2.fprop(), self._biases2.fprop())
            f_output = self._f.fprop(a2_output)
            valid_losses_this_epoch = self._f.fprop_loss(a2_output, valid_labels)

            # Classifications
            y_hat = np.argmax(f_output, axis=0)
            misclassification_error = hamming(valid_labels, y_hat)
            valid_losses.append(np.mean(valid_losses_this_epoch))
            valid_errors.append(misclassification_error)

            ## Train network
            for img, label in izip(train_imgs, train_labels):
                # Forward prop
                a0_output = self._a0.fprop(img, self._weights0.fprop(), self._biases0.fprop())
                h0_output = self._h0.fprop(a0_output)
                a1_output = self._a1.fprop(h0_output, self._weights1.fprop(), self._biases1.fprop())
                h1_output = self._h1.fprop(a1_output)
                a2_output = self._a2.fprop(h1_output, self._weights2.fprop(), self._biases2.fprop())
                f_output = self._f.fprop(a2_output)

                # Back prop gradients
                dl_da2 = self._f.bprop(label, f_output)
                dl_dW2 = self._weights2.bprop(dl_da2, h1_output)

                dl_db2 = self._biases2.bprop(dl_da2)
                dl_dh1 = self._h1.bprop(dl_da2, self._a2.weights.fprop())
                dl_da1 = self._a1.bprop(dl_dh1, a1_output)

                dl_dW1 = self._weights1.bprop(dl_da1, h0_output)
                dl_db1 = self._biases1.bprop(dl_da1)
                dl_dh0 = self._h0.bprop(dl_da1, self._a1.weights.fprop())
                dl_da0 = self._a0.bprop(dl_dh0, a0_output)

                dl_dW0 = self._weights0.bprop(dl_da0, img)
                dl_db0 = self._biases0.bprop(dl_da0)

                # SGD
                self._weights0.update(eta * dl_dW0.T)
                self._biases0.update(eta * dl_db0)
                self._weights1.update(eta * dl_dW1.T)
                self._biases1.update(eta * dl_db1)
                self._weights2.update(eta * dl_dW2.T)
                self._biases2.update(eta * dl_db2)

        return train_losses, valid_losses, train_errors, valid_errors, self._weights0.fprop()

if __name__ == '__main__':
    main()  # cProfile.run('main()')