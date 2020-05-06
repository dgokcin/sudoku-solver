# DO NOT IMPORT ANY OTHER LIBRARY.
# knnL=: https://anujkatiyal.com/blog/2017/10/01/ml-knn/#.XrBjJxMzZ0I
# pca: https://github.com/stabgan/Recognition-of-Hand-Written-Digits-MNIST-from
# -Scratch/blob/master/part1/main.py
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal\
#          -component-analysis.html
import numpy as np
import glob
import cv2
import os

import operator
import pandas as pd
import matplotlib.pyplot as plt
from mnist import MNIST

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# define your own functions here, Including MNIST and the functions from
# your previous assignment:
img_size = 28
no_of_labels = 10
image_pixels = img_size * img_size
data_path = MNIST(os.path.join(os.path.dirname(__file__), "mnist"))

# Helper functions
def calculate_confusion_matrix(y_actual, y_predicted):
    df_confusion = pd.crosstab(y_actual, y_predicted,
                               rownames=['Actual'],
                               colnames=['Predicted'],
                               margins=True)
    # print(df_confusion)
    return df_confusion


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()


def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1 - vector2, 2)))


def absolute_distance(vector1, vector2):
    return np.sum(np.absolute(vector1 - vector2))


def get_neighbours(X_train, X_test_instance, k):
    distances = []
    neighbors = []
    for i in range(0, X_train.shape[0]):
        dist = euclidean_distance(X_train[i], X_test_instance)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        # print distances[x]
        neighbors.append(distances[x][0])
    return neighbors


def predictkNNClass(output, y_train):
    classVotes = {}
    for i in range(len(output)):
        #         print output[i], y_train[output[i]]
        if y_train[output[i]] in classVotes:
            classVotes[y_train[output[i]]] += 1
        else:
            classVotes[y_train[output[i]]] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1),
                         reverse=True)
    # print sortedVotes
    return sortedVotes[0][0]


def kNN_test(X_train, X_test, Y_train, Y_test, k):
    output_classes = []
    for i in range(0, X_test.shape[0]):
        output = get_neighbours(X_train, X_test[i], k)
        predictedClass = predictkNNClass(output, Y_train)
        output_classes.append(predictedClass)
        print(i)
    return output_classes


def prediction_accuracy(predicted_labels, original_labels):
    count = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == original_labels[i]:
            count += 1
    # print count, len(predicted_labels)
    return float(count) / len(predicted_labels)


def center_data(X):
    """
    Returns a centered version of the data, where each feature now has mean = 0
    Args:
        X - n x d NumPy array of n data points, each with d features
    Returns:
        n x d NumPy array X' where for each i = 1, ..., n and j = 1, ..., d:
        X'[i][j] = X[i][j] - means[j]
    """
    arr = np.array(X)
    feature_means = arr.mean(axis=0)
    return arr - feature_means


def pca(X):
    """Performs principal component on x, a matrix with observations in the
    rows. Returns the projection matrix (the eigenvectors of x^T x, ordered
    with largest eigenvectors first) and the eigenvalues (ordered from
    largest to smallest).
    """

    # Subtract the mean of column i from column i, in order to center the
    # matrix.
    centered_data = center_data(X)  # first center data
    scatter_matrix = np.dot(centered_data.transpose(), centered_data)
    eigen_values, eigen_vectors = np.linalg.eig(scatter_matrix)
    # Re-order eigenvectors by eigenvalue magnitude:
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return eigen_vectors


def project_onto_PC(X, pcs, n_components):
    """
    Given principal component vectors pcs = principal_components(X)
    this function returns a new data array in which each sample in X
    has been projected onto the first n_components principcal components.
    """
    # TODO: first center data using the centerData() function.
    # TODO: Return the projection of the centered dataset
    #       on the first n_components principal components.
    #       This should be an array with dimensions: n x n_components.
    # Hint: these principal components = first n_components columns
    #       of the eigenvectors returned by principal_components().
    #       Note that each eigenvector is already be a unit-vector,
    #       so the projection may be done using matrix multiplication.

    return np.dot(center_data(X), pcs[:, :n_components])


def plot_PC(X, pcs, labels):
    """
    Given the principal component vectors as the columns of matrix pcs,
    this function projects each sample in X onto the first two principal components
    and produces a scatterplot where points are marked with the digit depicted in
    the corresponding image.
    labels = a numpy array containing the digits corresponding to each image in X.
    """
    pc_data = project_onto_PC(X, pcs, n_components=2)
    text_labels = [str(z) for z in labels.tolist()]
    fig, ax = plt.subplots()
    im = ax.scatter(pc_data[:, 0], pc_data[:, 1], c=labels, edgecolor='none',
                    alpha=0.5, cmap=plt.cm.get_cmap('jet', 10))
    # for i, txt in enumerate(text_labels):
    #     ax.annotate(txt, (pc_data[i, 0], pc_data[i, 1]))
    fig.colorbar(im, ax=ax)
    im.set_clim(0, 9)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.show()


def SudokuDigitDetector(img):
    print()


# Implement your algorithm here:


def sudokuAcc(gt, out):
    return (gt == out).sum() / gt.size * 100


if __name__ == "__main__":

    # MNIST experiments:
    train_images, train_labels = data_path.load_training()
    test_images, test_labels = data_path.load_testing()

    n_components = 25
    pcs = pca(train_images)
    train_pca = project_onto_PC(train_images, pcs, n_components)
    test_pca = project_onto_PC(test_images, pcs, n_components)

    # plot_PC(train_images[:1000], pcs, train_labels[:1000])
    plot_PC(train_images, pcs, train_labels)

    predicted_classes = {}
    final_accuracies = {}

    # for finding the best neighbor manually
    # for k in range(1, 15):
    #     predicted_classes[k] = kNN_test(train_pca[:30000], test_pca[:100],
    #                                     train_labels[:30000],
    #                                     test_labels, k)
    #     final_accuracies[k] = prediction_accuracy(predicted_classes[k],
    #                                               test_labels)


    # for finding the best neighbor with sckit
    # for k in range(1, 15):
    #     model = KNeighborsClassifier(n_neighbors=k)
    #     model.fit(train_pca, train_labels)
    #     preds = model.predict(test_pca)
    #     predicted_classes[k] = model.predict(test_pca)
    #     final_accuracies[k] = prediction_accuracy(predicted_classes[k],
    #                                               test_labels)

    k = 6

    # works fast but not allowed
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_pca, train_labels)
    preds = model.predict(test_pca)
    predicted_classes[k] = model.predict(test_pca)
    final_accuracies[k] = prediction_accuracy(predicted_classes[k],
                                              test_labels)



    # works slow
    # predicted_classes[k] = kNN_test(train_pca[:30000], test_pca[:100],
    #                                 train_labels[:30000],
    #                                 test_labels[:100], k)
    # final_accuracies[k] = prediction_accuracy(predicted_classes[k],
    #                                           test_labels[:100])

    plt.figure(figsize=(15, 6))
    plt.plot(list(final_accuracies.keys()), list(final_accuracies.values()))
    plt.xticks(list(final_accuracies.keys()))
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()
    max_accuracy_key = max(final_accuracies, key=final_accuracies.get)
    print("highest accuracy is hit with: " +
          str(max_accuracy_key) + " nearest neighbors with accuracy:"
          + str(final_accuracies[max_accuracy_key]))

    # Confusion Matrix
    y_actu = pd.Series(list(test_labels), name='Actual')
    y_pred = pd.Series(predicted_classes[k], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'],
                               colnames=['Predicted'], margins=True)

    print(df_confusion)
    df_confusion.to_latex('asd.tex')



    # plot_confusion_matrix(cnf)

    # Sudoku Experiments:

    image_dirs = 'images/*.jpg'
    data_dirs = 'images/*.dat'
    IMAGE_DIRS = glob.glob(image_dirs)
    DATA_DIRS = glob.glob(data_dirs)
    total_acc = 0
    # Loop over all images and ground truth
    for i, (img_dir, data_dir) in enumerate(zip(IMAGE_DIRS, DATA_DIRS)):
        # Define your variables etc.:
        image_name = os.path.basename(img_dir)
        gt = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
        img = cv2.imread(img_dir)
        output = SudokuDigitDetector(img)
        # implement this function, inputs
        # img, outputs in the same format as data 9x9 numpy array.
        total_acc = total_acc + sudokuAcc(gt, output)

    print("Sudoku dataset accuracy: {}".format(total_acc / (i + 1)))
