import numpy as np


class Regularization:
    def validate(self, layers_count):
        pass

    def reset(self, layers_count):
        pass

    def regularize_activation(self, layer, A):
        return A

    def regularize_cost(self, m_samples, W):
        return 0

    def regularize_weights(self, layer, m_samples, dW, W, db, b):
        return dW, db

    def regularize_derivative(self, layer, dA):
        return dA


class L2Regularization(Regularization):
    """
    Weight decay family regularization (also called Frobenius normalization when applied to neuron nets)
    Increase the "lambda" hyper-parameter to perform smaller steps
    """
    def __init__(self, lambd):
        self.lambd = lambd

    def regularize_cost(self, m_samples, W):
        L2_regularization_cost = 0
        if self.lambd > 0:
            for l in range(1, len(W)):
                L2_regularization_cost += np.sum(np.square(W[l]))
            L2_regularization_cost *= self.lambd / (2 * m_samples)
        return L2_regularization_cost

    def regularize_weights(self, layer, m_samples, dW, W, db, b):
        if self.lambd > 0:
            return dW + W * self.lambd / m_samples, db
        else:
            return dW, db


class DropOutRegularization(Regularization):
    """
    Randomly switches off neurons to prevent overfitting
    For each layer, decide the probability to keep neuron (from 0 to 1)
    Same same neuron must be switched off in back propagation
    """
    def __init__(self, layers_probs):
        assert (len(layers_probs) > 0)
        self.layers_probs = layers_probs
        self.D = []

    def validate(self, layers_count):
        assert (len(self.layers_probs) <= layers_count)

    def reset(self, layers_count):
        self.D = [None]*layers_count

    def regularize_activation(self, layer, A):
        if layer <= len(self.layers_probs):
            keep_prob = self.layers_probs[layer - 1]
            if keep_prob is not None and keep_prob < 1:
                D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
                A = np.multiply(A, D) / keep_prob
                self.D[layer - 1] = D
        return A

    def regularize_derivative(self, layer, dA):
        D = self.D[layer - 1]
        if D is not None:
            return np.multiply(dA, D)
        else:
            return dA


