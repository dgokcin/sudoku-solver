# Deniz Gokcin S021834 Department of Computer Science
# DO NOT IMPORT ANY OTHER LIBRARY.

import math

import numpy as np
import glob
import cv2
import os

import pandas as pd
import matplotlib.pyplot as plt
from mnist import MNIST

# define your own functions here, Including MNIST and the functions from
# your previous assignment:
img_size = 28
no_of_labels = 10
image_pixels = img_size * img_size
data_path = MNIST(os.path.join(os.path.dirname(__file__), "mnist"))


# Helper functions for mnist & pca
def calculate_confusion_matrix(y_actual, y_predicted):
    """
    Calculates the distance matrix when given actial values and predicted
    values
    """
    df_confusion = pd.crosstab(y_actual, y_predicted,
                               rownames=['Actual'],
                               colnames=['Predicted'],
                               margins=True)
    # print(df_confusion)
    return df_confusion


def compute_distances(X, X_train):
    """
    Compute the distance between each test point in X and each training point
    in X_train using no loops
    """
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.reshape(np.sum(X**2, axis=1), [num_test,1]) + np.sum(X_train**2, axis=1) \
            - 2 * np.matmul(X, X_train.T)
    dists = np.sqrt(dists)
    return dists


def predict_labels(dists, y_train, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = []
        closest_y = y_train[np.argsort(dists[i])][0:k]
        y_pred[i] = np.bincount(closest_y).argmax()

    return y_pred


def center_data(X):
    """
    Returns a centered version of the data, where each feature now has mean = 0
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
    Given principal component vectors eigen_vectors, this function returns
    a new data array in which each sample in X has been projected onto the
    first n_components principcal components.
    """
    return np.dot(center_data(X), pcs[:, :n_components])


def plot_PC(X, pcs, labels):
    """
    Given the principal component vectors as the columns of matrix
    eigen_vectors, this function projects each sample in X onto the first
    two principal components.
    """
    pc_data = project_onto_PC(X, pcs, n_components=2)
    fig, ax = plt.subplots()
    im = ax.scatter(pc_data[:, 0], pc_data[:, 1], c=labels, edgecolor='none',
                    alpha=0.5, cmap=plt.cm.get_cmap('jet', 10))
    fig.colorbar(im, ax=ax)
    im.set_clim(0, 9)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.show()


def reconstruct_PC(x_pca, pcs, n_components, X):
    """
    Given the principal component vectors as the columns of matrix pcs,
    this function reconstructs a single image from its principal component
    representation, x_pca.
    X = the original data to which PCA was applied to get pcs.
    """
    feature_means = X - center_data(X)
    feature_means = feature_means[0, :]
    x_reconstructed = np.dot(x_pca, pcs[:, range(n_components)].T) + feature_means
    return x_reconstructed


def plot_images(X):
    if X.ndim == 1:
        X = np.array([X])
    num_images = X.shape[0]
    num_rows = math.floor(math.sqrt(num_images))
    num_cols = math.ceil(num_images/num_rows)
    for i in range(num_images):
        reshaped_image = X[i,:].reshape(28,28)
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(reshaped_image, cmap=plt.cm.get_cmap('Greys_r', 10))
        plt.axis('off')
    plt.show()


# Helper functions for sudoku stuff
# Plots the original image next to the processed image
def plot_original_final(original, final):
    plt.subplot(121), plt.imshow(original, 'gray'), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(final, 'gray'), plt.title('final')
    plt.xticks([]), plt.yticks([])
    plt.show()


# Calculates the distance two points.
def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
    """Crops and warps a rectangular section from an image into
     a square of similar size."""

    # Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Get the longest side in the rectangle
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    # Describe a square with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    return cv2.warpPerspective(img, m, (int(side), int(side)))


def get_corners(img):
    """
    This function extracts and returns the corner points of the largest
    contours
    """
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                      reverse=True)
    largest_contour = np.squeeze(contours[0])

    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right],
               largest_contour[bottom_left],
               largest_contour[bottom_right]]
    return corners


def draw_corners(input_image, corners, radius=15, colour=(255, 0, 0)):
    """
    When given an image and corner points, this function draws the corners
    onto that image.
    """
    img = input_image.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in corners:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)

    return img


def merge_empty_cells_with_predictions(grid, predictions):
    """
    This function merges the detected empty cells and the predictions with
    knn into one 9 x 9 numpy array
    """
    predictions = np.reshape(predictions, (-1, 9))
    mask = (grid == -1)
    new_array = np.copy(grid)
    new_array[mask] = predictions[mask]
    return new_array


def SudokuDigitDetector(img, pcs, X_train):

    # Copy the original image
    original = img.copy()

    # Convert to grayscale, since color information is not needed.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gausian Blur to remove noise a little
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # Apply adaptive threshold to seperate the foreground from the
    # background
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 31, 2)


    # Find all contours in the threshold image.
    contours, hire = cv2.findContours(thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    # Assumption: The biggest contour in the image is the sudoku grid.
    contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                      reverse=True)
    largest_contour = np.squeeze(contours[0])

    # Create a mask image so that we ignore everything else outside the
    # grid and focus on the board
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    cv2.drawContours(mask, [largest_contour], 0, 0, 2)

    # Extract the region of interest using the mask generated above.
    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]

    # Apply Gausian Blur to remove noise a little(to region of interest)
    blur_roi = cv2.GaussianBlur(out, (15, 15), 0)

    # Apply adaptive threshold.
    thresh_roi = cv2.adaptiveThreshold(blur_roi, 255, 1, 1, 11, 2)

    # Detect the cornerpoints of the thresholded image
    corners = get_corners(thresh_roi)

    # Apply 4-point transform and wrap
    cropped = crop_and_warp(gray, corners)
    blurred = cv2.GaussianBlur(cropped, (15, 15), 0)
    thr = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    # Prepare the image for pca
    cv2.bitwise_not(thr, thr)
    # plot_original_final(original, thr)
    # cropped_thresh = crop_and_warp(thresh_roi, corners)

    sudoku = cv2.resize(thr, (252, 252))
    grid = np.zeros([9, 9])
    pieces = []
    for i in range(9):
        for j in range(9):
            image = sudoku[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28]
            # If mean is smaller than 45 this is an empty cell.
            if image.mean() < 45:
                grid[i][j] = 0
            else:
                grid[i][j] = -1

            # Crop 3 Pixels from each side to remove grid lines.
            cropped = image[3:25, 3:25]
            re_resized = cv2.resize(cropped, (28, 28))
            # plot_original_final(cropped, re_resized)
            # plot_original_final(sudoku, re_resized)
            flatten_piece = re_resized.flatten()
            pieces.append(project_onto_PC(flatten_piece, pcs,
                                          n_components))
    grid = grid.astype(int)

    # Reduct image from its principal components
    # reducted_image = reconstruct_PC(pieces[0], pcs,
    #                                 n_components, train_images)
    # plot_images(reducted_image)

    pieces = np.array(pieces)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    dists = compute_distances(pieces, X_train)

    predictions = predict_labels(dists, y_train, k=6)
    output = merge_empty_cells_with_predictions(grid, predictions)
    return output


def sudokuAcc(gt, out):
    return (gt == out).sum() / gt.size * 100


if __name__ == "__main__":

    # MNIST experiments:
    train_images, train_labels = data_path.load_training()
    test_images, test_labels = data_path.load_testing()

    n_components = 25
    eigen_vectors = pca(train_images)

    # Dimensionality reduction
    X_train = project_onto_PC(train_images, eigen_vectors, n_components)
    X_test = project_onto_PC(test_images, eigen_vectors, n_components)


    # For visualization
    # original_image = np.asarray(train_images[0])
    # reducted_image = reconstruct_PC(X_train[0], eigen_vectors,
    #                                 n_components, train_images)
    # plot_images(original_image)
    # plot_images(reducted_image)

    # Convert the labels to numpy arrays
    y_train = np.asarray(train_labels)
    y_test = np.asarray(test_labels)

    # Fast plot onto 2 components.
    # plot_PC(train_images, eigen_vectors, y_train)


    # Mask for debugging purposes...
    num_training = 60000
    # num_training = 500
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 10000
    # num_test = 50
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # # Compute the distances and store them in dists.
    print('Computing Distances...')
    dists = compute_distances(X_test, X_train)

    final_accuracies = {}
    predicted_classes = {}
    # ##########################################################################
    # # for discovering the best k, should be done once
    # # for k in range(1, 15):
    # #     y_test_pred = predict_labels(dists, y_train, k=k)
    # #     num_correct = np.sum(y_test_pred == y_test)
    # #     accuracy = float(num_correct) / num_test
    # #
    # #     final_accuracies[k] = accuracy
    # #     predicted_classes[k] = y_test_pred
    # #
    # #     print('With %d neighbours, Got %d / %d correct => accuracy: %f' % (
    # #         k, num_correct, num_test, accuracy))
    # #
    # # # Plot the accuracy vs k graph
    # # plt.figure(figsize=(15, 6))
    # # plt.plot(list(final_accuracies.keys()), list(final_accuracies.values()))
    # # plt.xticks(list(final_accuracies.keys()))
    # # plt.xlabel("k")
    # # plt.ylabel("Accuracy")
    # # plt.show()
    # ##########################################################################

    # Hardcoded best k value for 25 components
    k = 6

    y_test_pred = predict_labels(dists, y_train, k=k)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test

    final_accuracies[k] = accuracy
    predicted_classes[k] = y_test_pred

    print('With %d neighbours, Got %d / %d correct => accuracy: %f' % (
        k, num_correct, num_test, accuracy))
    # ##########################################################################


    max_accuracy_key = max(final_accuracies, key=final_accuracies.get)


    # Confusion Matrix for MNIST
    y_actu = pd.Series(list(y_test), name='Actual')
    y_pred = pd.Series(predicted_classes[max_accuracy_key], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'],
                               colnames=['Predicted'], margins=True)


    print(df_confusion)

    # Save the confusion matrix as latex to include in the report
    df_confusion.to_latex('confusion_matrix.tex')

    # Sudoku Experiments:
    image_dirs = 'images/*.jpg'
    data_dirs = 'images/*.dat'
    # image_dirs = 'v2_test/*.jpg'
    # data_dirs = 'v2_test/*.dat'
    # image_dirs = 'playground/*.jpg'
    # data_dirs = 'playground/*.dat'
    IMAGE_DIRS = glob.glob(image_dirs)
    DATA_DIRS = glob.glob(data_dirs)
    total_acc = 0
    print('Sudoku Detection has started')
    # Loop over all images and ground truth
    for i, (img_dir, data_dir) in enumerate(zip(IMAGE_DIRS, DATA_DIRS)):
        # Define your variables etc.:
        image_name = os.path.basename(img_dir)
        gt = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
        img = cv2.imread(img_dir)
        output = SudokuDigitDetector(img, eigen_vectors, X_train)
        # implement this function, inputs
        # img, outputs in the same format as data 9x9 numpy array.
        total_acc = total_acc + sudokuAcc(gt, output)

    print("Sudoku dataset accuracy: {}".format(total_acc / (i + 1)))
