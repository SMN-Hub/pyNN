import numpy as np
from deeplearn.activation import ACTIVATIONS, BACKWARD_DERIVATIONS
from deeplearn.regularization import Regularization


class NeuralNetLearn:
    def __init__(self, layers_sizes):
        self.layers_sizes = layers_sizes
        self.hidden_layer_activation = 'relu'
        self.output_layer_activation = 'sigmoid'
        self.print_cost_rate = 0
        self.regularizations = []
        self.cache = {}
        self.__reset()

    def __reset(self):
        self.cache = {'W': [], 'b': [], 'A': [], 'Z': [], 'dW': [], 'db': []}
        for reg in self.regularizations:
            reg.reset()

    def get_cache(self, kind, layer):
        return self.cache[kind][layer - 1]

    def set_cache(self, kind, layer, value):
        self.cache[kind].insert(layer - 1, value)

    def add_regularization(self, regularization:Regularization):
        regularization.validate(len(self.layers_sizes))
        self.regularizations.append(regularization)

    def regularize_activation(self, layer, A):
        for reg in self.regularizations:
            A = reg.regularize_activation(layer, A)
        return A

    def regularize_cost(self, m_samples, W):
        regularization_cost = 0
        for reg in self.regularizations:
            regularization_cost += reg.regularize_cost(m_samples, W)
        return regularization_cost

    def regularize_weights(self, m_samples, dW, W):
        for reg in self.regularizations:
            dW = reg.regularize_weights(m_samples, dW, W)
        return dW

    def regularize_derivative(self, layer, dA):
        for reg in self.regularizations:
            dA = reg.regularize_derivative(layer, dA)
        return dA

    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (tuple) containing the dimensions of each layer in our network
        """
        np.random.seed(1)
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            self.set_cache('W', l, np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1]))
            self.set_cache('b', l, np.zeros((layer_dims[l], 1)))

    def linear_activation_forward(self, layer, A_prev, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        layer -- current layer
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation function to be used in this layer

        Returns:
        A -- the output of the activation function, also called the post-activation value
        """

        # weights matrix
        W = self.get_cache('W', layer)
        # bias vector
        b = self.get_cache('b', layer)

        Z = W.dot(A_prev) + b
        A = activation(Z)

        A = self.regularize_activation(layer, A)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))

        self.set_cache('Z', layer, Z)
        self.set_cache('A', layer, A)

        return A

    def forward_propagation(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)

        Returns:
        AL -- last post-activation value
        """

        A = X
        L = len(self.cache['W'])  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1).
        for layer in range(1, L):
            A_prev = A
            A = self.linear_activation_forward(layer, A_prev, ACTIVATIONS[self.hidden_layer_activation])

        # Implement LINEAR -> SIGMOID.
        AL = self.linear_activation_forward(L, A, ACTIVATIONS[self.output_layer_activation])

        assert (AL.shape == (1, X.shape[1]))

        return AL

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        cross_entropy_cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
        cross_entropy_cost = np.squeeze(cross_entropy_cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cross_entropy_cost.shape == ())

        return cross_entropy_cost + self.regularize_cost(m, self.cache['W'])

    def linear_activation_backward(self, layer, dA, activation_backward):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation_backward -- the activation backward function

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        """
        A_prev = self.get_cache('A', layer - 1)
        W = self.get_cache('W', layer)
        b = self.get_cache('b', layer)
        Z = self.get_cache('Z', layer)

        dZ = activation_backward(dA, Z)

        m = A_prev.shape[1]

        dW = 1. / m * np.dot(dZ, A_prev.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        # Regularization
        dW = self.regularize_weights(m, dW, W)
        dA_prev = self.regularize_derivative(layer, dA_prev)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        # cache
        self.set_cache('dW', layer, dW)
        self.set_cache('db', layer, db)

        return dA_prev

    def backward_propagation(self, AL, Y):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        """
        L = len(self.cache['W'])  # the number of layers
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients.
        dA = self.linear_activation_backward(L, dAL, BACKWARD_DERIVATIONS[self.output_layer_activation])

        for l in reversed(range(1, L)):
            # lth layer: (RELU -> LINEAR) gradients.
            dA = self.linear_activation_backward(l, dA, BACKWARD_DERIVATIONS[self.hidden_layer_activation])

    def fit(self, X, Y, learning_rate=0.0075, max_iter=2500):
        self.__reset()
        costs = []  # keep track of cost

        n_x = X.shape[0]
        layers_dims = tuple(n_x) + self.layers_sizes

        # Parameters initialization.
        self.initialize_parameters_deep(layers_dims)

        # Loop (gradient descent)
        for i in range(0, max_iter):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL = self.forward_propagation(X)

            # Compute cost.
            cost = self.compute_cost(AL, Y)

            # Backward propagation.
            self.backward_propagation(AL, Y)

            # Update parameters.
            self.update_parameters(learning_rate)

            # Print the cost every 100 training example
            if self.print_cost_rate > 0 and i % self.print_cost_rate == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                costs.append(cost)


