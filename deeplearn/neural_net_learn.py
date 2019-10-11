import numpy as np
import matplotlib.pyplot as plt
from deeplearn.activation import ACTIVATIONS, BACKWARD_DERIVATIONS
from deeplearn.regularization import Regularization
from deeplearn.dataset_utils import slice_data, shuffle_data
from deeplearn.learning_rate_decay import LearningRate

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


class NeuralNetLearn:
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    """
    def __init__(self, layers_sizes, init_shuffle_seed=1, data_shuffle_seed=None):
        """
        Arguments:
        layers_sizes -- list containing the input size and each layer size.
        init_shuffle_seed -- Random seed for shuffling data (mandatory).
        data_shuffle_seed -- Random seed for shuffling data (optional, None for no shuffle).
        """
        self.layers_sizes = layers_sizes
        self.init_shuffle_seed = init_shuffle_seed
        self.data_shuffle_seed = data_shuffle_seed
        self.batch_threshold = None
        self.batch_size = None
        self.hidden_layer_activation = 'relu'
        self.output_layer_activation = 'sigmoid'
        self.hidden_layer_init_factor = 1
        self.output_layer_init_factor = 1
        self.print_cost_rate = 100
        self.__regularizations = []
        self.__cache = {}
        self.__reset(True)

    def use_mini_batch(self, batch_size=512, batch_threshold=2_000):
        """
        Activate mini batch slicing

        Arguments:
        batch_size -- Mini batch size, triggered if data size exceeds batch_threshold.
        batch_threshold -- Data size threshold to trigger mini batch slice (for small data sets < 2000, mini batch is generally not necessary).
        """
        self.batch_threshold = batch_threshold
        self.batch_size = batch_size

    def __reset(self, full: bool):
        size = len(self.layers_sizes) + 1
        if full:  # complete reset
            self.__cache = {'W': [None] * size, 'b': [None] * size, 'A': [None] * size, 'Z': [None] * size, 'dW': [None] * size, 'db': [None] * size}
        else:  # inter-batch reset: keep W & b
            self.__cache.update({'A': [None] * size, 'Z': [None] * size, 'dW': [None] * size, 'db': [None] * size})
        for reg in self.__regularizations:
            reg.reset(size-1)

    @property
    def parameters(self):
        return self.__cache['W'], self.__cache['b']

    def get_cache(self, kind, layer):
        return self.__cache[kind][layer]

    def set_cache(self, kind, layer, value):
        self.__cache[kind][layer] = value

    def add_regularization(self, regularization: Regularization):
        regularization.validate(len(self.layers_sizes))
        self.__regularizations.append(regularization)

    def _regularize_activation(self, layer, A):
        for reg in self.__regularizations:
            A = reg.regularize_activation(layer, A)
        return A

    def _regularize_cost(self, m_samples, W):
        regularization_cost = 0
        for reg in self.__regularizations:
            regularization_cost += reg.regularize_cost(m_samples, W)
        return regularization_cost

    def _regularize_weights(self, layer, m_samples, dW, W, db, b):
        for reg in self.__regularizations:
            dW, db = reg.regularize_weights(layer, m_samples, dW, W, db, b)
        return dW, db

    def _regularize_derivative(self, layer, dA):
        for reg in self.__regularizations:
            dA = reg.regularize_derivative(layer, dA)
        return dA

    def _initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (tuple) containing the dimensions of each layer in our network, including input (layer 0)
        """
        np.random.seed(self.init_shuffle_seed)
        L = len(layer_dims)  # number of layers in the network
        # Hidden layers
        for layer in range(1, L-1):
            self._initialize_layer_parameters(layer_dims, layer, self.hidden_layer_init_factor)
        # Output layer
        self._initialize_layer_parameters(layer_dims, L-1, self.output_layer_init_factor)

    def _initialize_layer_parameters(self, layer_dims, layer, factor):
        self.set_cache('W', layer, np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * np.sqrt(factor / layer_dims[layer - 1]))
        self.set_cache('b', layer, np.zeros((layer_dims[layer], 1)))

    def _linear_activation_forward(self, layer, A_prev, activation, with_regularization):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        layer -- current layer
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation function to be used in this layer
        with_regularization -- apply regularization (for training only)

        Returns:
        A -- the output of the activation function, also called the post-activation value
        """

        # weights matrix
        W = self.get_cache('W', layer)
        # bias vector
        b = self.get_cache('b', layer)

        Z = W.dot(A_prev) + b
        A = activation(Z)

        if with_regularization:
            A = self._regularize_activation(layer, A)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))

        self.set_cache('Z', layer, Z)
        self.set_cache('A', layer, A)

        return A

    def _forward_propagation(self, X, with_regularization):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        with_regularization -- apply regularization (for training only)

        Returns:
        AL -- last post-activation value
        """

        self.set_cache('A', 0, X)
        A = X
        L = len(self.layers_sizes)  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1).
        for layer in range(1, L):
            A_prev = A
            A = self._linear_activation_forward(layer, A_prev, ACTIVATIONS[self.hidden_layer_activation], with_regularization)

        # Implement LINEAR -> SIGMOID.
        AL = self._linear_activation_forward(L, A, ACTIVATIONS[self.output_layer_activation], with_regularization)

        assert (AL.shape == (1, X.shape[1]))

        return AL

    def _compute_cost(self, AL, Y):
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
        cross_entropy_cost = -np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T)
        cross_entropy_cost = np.squeeze(cross_entropy_cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cross_entropy_cost.shape == ())

        return cross_entropy_cost + self._regularize_cost(m, self.__cache['W'])

    def _linear_activation_backward(self, layer, dA, activation_backward):
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
        # Regularization
        dW, db = self._regularize_weights(layer, m, dW, W, db, b)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        dA_prev = None
        if layer > 1:  # no use to compute dA for layer 1
            dA_prev = np.dot(W.T, dZ)
            # Regularization
            dA_prev = self._regularize_derivative(layer - 1, dA_prev)
            assert (dA_prev.shape == A_prev.shape)

        # cache
        self.set_cache('dW', layer, dW)
        self.set_cache('db', layer, db)

        return dA_prev

    def _backward_propagation(self, AL, Y):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        """
        L = len(self.layers_sizes)  # the number of layers
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients.
        dA = self._linear_activation_backward(L, dAL, BACKWARD_DERIVATIONS[self.output_layer_activation])

        for layer in reversed(range(1, L)):
            # lth layer: (RELU -> LINEAR) gradients.
            dA = self._linear_activation_backward(layer, dA, BACKWARD_DERIVATIONS[self.hidden_layer_activation])

    def _update_parameters(self, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        learning_rate -- alpha hyper parameter
        """
        W = self.__cache['W']
        b = self.__cache['b']
        dW = self.__cache['dW']
        db = self.__cache['db']

        L = len(self.layers_sizes)  # number of layers in the neural network

        # Update rule for each parameter.
        upd = slice(1, L+1)
        W[upd] = W[upd] - np.multiply(learning_rate, dW[upd])
        b[upd] = b[upd] - np.multiply(learning_rate, db[upd])

    def _fit_batch(self, X, Y, learning_rate):
        """
        Trains a L-layer neural network batch

        Arguments:
        X -- data, numpy array of shape (number of features, number of examples)
        Y -- true "label" vector, of shape (1, number of examples)
        learning_rate -- learning rate of the gradient descent update rule
        """
        self.__reset(False)

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL = self._forward_propagation(X, True)

        # Compute cost.
        cost = self._compute_cost(AL, Y)

        # Backward propagation.
        self._backward_propagation(AL, Y)

        # Update parameters.
        self._update_parameters(learning_rate)

        return cost

    def fit(self, X, Y, learning_rate_f: LearningRate, max_iter=2500, plot_costs=True):
        """
        Trains a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (number of features, number of examples)
        Y -- true "label" vector, of shape (1, number of examples)
        learning_rate -- learning rate of the gradient descent update rule
        max_iter -- number of iterations of the optimization loop
        """
        self.__reset(True)
        costs = []  # keep track of cost

        n_x = X.shape[0]
        m_samples = X.shape[1]
        learning_rate_zero = learning_rate_f.next(0, 0)
        learning_rate = learning_rate_zero
        epoch = 0
        seed = self.data_shuffle_seed
        train_X, train_Y = X, Y

        # Parameters initialization.
        layers_dims = (n_x,) + self.layers_sizes
        self._initialize_parameters_deep(layers_dims)

        # Loop (gradient descent)
        for i in range(0, max_iter):
            # Shuffle data if configured
            if seed is not None:
                seed += 1
                train_X, train_Y = shuffle_data(X, Y, seed)
            # update learning rate
            learning_rate = learning_rate_f.next(i, epoch)
            cumulative_loss = 0
            # Loop (mini batch)
            for batch_slice in slice_data(m_samples, self.batch_size, self.batch_threshold):
                # Train BATCH
                batch_cost = self._fit_batch(train_X[:, batch_slice], train_Y[:, batch_slice], learning_rate)
                cumulative_loss += batch_cost
            cost = cumulative_loss / m_samples
            # Print the cost every 100 training example
            if self.print_cost_rate > 0 and i % self.print_cost_rate == 0:
                epoch += 1
                print(f"Cost after iteration {i}: {cost} - learning rate: {learning_rate}")
                costs.append(cost)
        # plot the cost
        if plot_costs and self.print_cost_rate > 0:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel(f"iterations (per {self.print_cost_rate})")
            if learning_rate_zero != learning_rate:
                plt.title(f"Learning rate = [{learning_rate_zero} : {learning_rate}]")
            else:
                plt.title(f"Learning rate = {learning_rate}")
            plt.show()

    def predict(self, X, y=None):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label, , numpy array of shape (number of features, number of examples)
        Y -- true "label" vector, of shape (1, number of examples)
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        p = np.zeros((1, m))

        # Forward propagation
        probas = self._forward_propagation(X, False)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1

        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        if y is not None:
            print("Accuracy: " + str(np.mean((p[0,:] == y[0,:]))))

        return p
