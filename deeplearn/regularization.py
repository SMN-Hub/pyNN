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


class VelocityRegularization(Regularization):
    """
    Speed up convergence using weighted velocity
    """
    def __init__(self, beta):
        self.beta = beta
        self.dW_velocity = None
        self.db_velocity = None

    def reset(self, layers_count):
        self.dW_velocity = [0.]*layers_count
        self.db_velocity = [0.]*layers_count

    def update_velocity(self, layer, dW, db):
        dW_velocity = self.beta * self.dW_velocity[layer-1] + (1-self.beta) * dW
        db_velocity = self.beta * self.db_velocity[layer-1] + (1-self.beta) * db
        self.dW_velocity[layer-1] = dW_velocity
        self.db_velocity[layer-1] = db_velocity
        return dW_velocity, db_velocity


class MomentumRegularization(VelocityRegularization):
    """
    Mementum allow to damp out the oscillations in gradient descent and increase learning rate
    Use exponentially weighted average or root mean square
    """
    def __init__(self, beta=0.9):
        super().__init__(beta)

    def regularize_weights(self, layer, m_samples, dW, W, db, b):
        return self.update_velocity(layer, dW, db)


class RMSpropRegularization(VelocityRegularization):
    """
    RMS prop allow to increase learning rate
    Use root mean square
    """
    def __init__(self, beta=0.99, epsilon=1e-6):
        super().__init__(beta)
        self.epsilon = epsilon

    def update_velocity(self, layer, dW, db):
        return super().update_velocity(layer, np.power(dW, 2), np.power(db, 2))

    def regularize_weights(self, layer, m_samples, dW, W, db, b):
        dW_velocity, db_velocity = self.update_velocity(layer, dW, db)
        return self.scale(dW, dW_velocity), self.scale(db, db_velocity)

    def scale(self, theta, velocity):
        return np.divide(theta, np.sqrt(velocity) + self.epsilon)


class AdamOptimization(Regularization):
    """
    Adaptive moment estimation
    Use Momentum + RMS
    """
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.momentum = MomentumRegularization(beta1)
        self.rmsprop = RMSpropRegularization(beta2, epsilon)

    def reset(self, layers_count):
        self.momentum.reset(layers_count)
        self.rmsprop.reset(layers_count)

    def regularize_weights(self, layer, m_samples, dW, W, db, b):
        # Compute velocity + squared
        vdW, vdb = self.momentum.update_velocity(layer, dW, db)
        sdW, sdb = self.rmsprop.update_velocity(layer, dW, db)
        # Rectify derivative
        vdW /= 1 - self.momentum.beta
        vdb /= 1 - self.momentum.beta
        sdW /= 1 - self.rmsprop.beta
        sdb /= 1 - self.rmsprop.beta
        # Optimize gradients
        return self.rmsprop.scale(vdW, sdW), self.rmsprop.scale(vdb, sdb)
