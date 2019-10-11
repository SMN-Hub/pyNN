import numpy as np
from deeplearn.regularization import Regularization


class VelocityOptimization(Regularization):
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


class MomentumOptimization(VelocityOptimization):
    """
    Mementum allow to damp out the oscillations in gradient descent and increase learning rate
    Use exponentially weighted average or root mean square
    """
    def __init__(self, beta=0.9):
        super().__init__(beta)

    def regularize_weights(self, layer, m_samples, dW, W, db, b):
        return self.update_velocity(layer, dW, db)


class RMSpropOptimization(VelocityOptimization):
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
        self.momentum = MomentumOptimization(beta1)
        self.rmsprop = RMSpropOptimization(beta2, epsilon)

    def reset(self, layers_count):
        self.momentum.reset(layers_count)
        self.rmsprop.reset(layers_count)

    def regularize_weights(self, layer, m_samples, dW, W, db, b):
        # Compute velocity + squared
        vdW, vdb = self.momentum.update_velocity(layer, dW, db)
        sdW, sdb = self.rmsprop.update_velocity(layer, dW, db)
        # Bias correction
        vdW /= 1 - (self.momentum.beta ** 2)
        vdb /= 1 - (self.momentum.beta ** 2)
        sdW /= 1 - (self.rmsprop.beta ** 2)
        sdb /= 1 - (self.rmsprop.beta ** 2)
        # Optimize gradients
        return self.rmsprop.scale(vdW, sdW), self.rmsprop.scale(vdb, sdb)
