from sklearn.neural_network import MLPClassifier

from cat_training import load_data


def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1)  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1)

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    mlp = MLPClassifier(hidden_layer_sizes=(20, 7, 5), verbose=0, random_state=0, max_iter=2500, solver='adam', learning_rate_init=0.0075)
    mlp.fit(train_x, train_y.T)
    print("Training set score: %f" % mlp.score(train_x, train_y.T))
    print("Training set loss: %f" % mlp.loss_)

    print("Test set score: %f" % mlp.score(test_x, test_y.T))


if __name__ == "__main__":
    main()
