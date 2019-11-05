import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import deeplearn

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def load_data(plot=True):
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
    # Visualize the data
    if plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
        plt.show()
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    return train_X, train_Y


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def main():
    train_X, train_Y = load_data(False)
    print("train_x's shape: " + str(train_X.shape))
    print("train_y's shape: " + str(train_Y.shape))
    # simple gradient descent
    layers_dims = (5, 2, 1)  # 3-layer model
    # Train the model
    classifier = deeplearn.NeuralNetLearn(layers_dims, init_shuffle_seed=3, data_shuffle_seed=10)
    classifier.use_mini_batch(batch_size=64, batch_threshold=0)
    classifier.print_cost_rate = 1000
    # classifier.output_layer_init_factor = 2
    # classifier.hidden_layer_init_factor = 2
    classifier.add_regularization(deeplearn.AdamOptimization(0.9))
    learning_rate = deeplearn.LearningRateDecayLinear(0.0007, 0)
    classifier.fit(train_X, train_Y, learning_rate, max_iter=10_000, plot_costs=False)
    classifier.predict(train_X, train_Y)

    # Plot decision boundary
    # plt.title("Model with Gradient Descent optimization")
    # axes = plt.gca()
    # axes.set_xlim([-1.5,2.5])
    # axes.set_ylim([-1,1.5])
    # plot_decision_boundary(lambda x: (classifier.predict(x.T) == 1), train_X, train_Y)


if __name__ == "__main__":
    main()
