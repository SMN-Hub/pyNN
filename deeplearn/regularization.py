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

    def regularize_weights(self, m_samples, dW, W):
        return dW

    def regularize_derivative(self, layer, dA):
        return dA


class L2Regularization(Regularization):
    def __init__(self, lambd):
        self.lambd = lambd

    def regularize_cost(self, m_samples, W):
        L2_regularization_cost = 0
        if self.lambd > 0:
            for l in range(1, len(W)):
                L2_regularization_cost += np.sum(np.square(W[l]))
            L2_regularization_cost *= self.lambd / (2 * m_samples)
        return L2_regularization_cost

    def regularize_weights(self, m_samples, dW, W):
        if self.lambd > 0:
            return dW + W * self.lambd / m_samples
        else:
            return dW


class DropOutRegularization(Regularization):
    def __init__(self, layers_probs):
        assert (len(layers_probs) > 0)
        self.layers_probs = layers_probs
        self.D = []

    def validate(self, layers_count):
        assert (len(self.layers_probs) <= layers_count)

    def reset(self, layers_count):
        self.D = [None]*layers_count

    def regularize_activation(self, layer, A):
        D = None
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
