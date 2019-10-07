import numpy as np
import matplotlib.pyplot as plt
import h5py
import deeplearn

np.random.seed(1)
plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    index = 10
    # plt.imshow(train_x_orig[index])
    # plt.show()
    print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

    # Explore your dataset
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    n_x = train_x.shape[0]  # num_px * num_px * 3

    layers_dims = (20, 7, 5, 1)  # 4-layer model
    # Train the model
    classifier = deeplearn.NeuralNetLearn(layers_dims)
    classifier.add_regularization(deeplearn.L2Regularization(0.7))
    classifier.add_regularization(deeplearn.DropOutRegularization((0.7, 0.8, 0.9, 1)))
    classifier.fit(train_x, train_y)

    # Predict result with trained parameters
    print("On the train set:")
    pred_train = classifier.predict(train_x, train_y)
    print("On the test set:")
    pred_test = classifier.predict(test_x, test_y)


if __name__ == "__main__":
    main()
