# Deniz Gokcin S021834 Department of Computer Science
import cv2
import os, fnmatch
import numpy as np
from matplotlib import pyplot as plt


#   Extracts all the .jpg files in a directory and returns it as a list.
def get_list_of_files(directory_name):
    jpg_files = []
    all_files = os.listdir(directory_name)
    pattern = "*.jpg"
    for entry in all_files:
        if fnmatch.fnmatch(entry, pattern):
            jpg_files.append(os.path.join(directory_name, entry))

    return jpg_files


#   This method plots the obtained imabes during the execution of the algorithm
def plot_images(img1, img2, img3, title1='', title2='', title3=''):

    plt.subplot(131), plt.imshow(img1), plt.title(title1)
    plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(img2, cmap='gray'), plt.title(title2)
    plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(img3, cmap='gray'), plt.title(title3)
    plt.xticks([]), plt.yticks([])

    plt.show()


#   Plots a single image
def plot_single_image(img, title=''):
    plt.subplot(111), plt.imshow(img), plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()


#   Plots the original image next to the processed image
def plot_original_final(original, final):
    plt.subplot(121), plt.imshow(original), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(final), plt.title('final')
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    # images = get_list_of_files('v2_test')
    images = get_list_of_files('playground')
    # images = ['playground/image1019.jpg', 'playground/image18.jpg']
    # images = ['playground/image1019.jpg']

    for image in images:
        image = cv2.imread(image)
        original = image.copy()

        # Convert to grayscale, since color information is not needed.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gausian Blur to remove noise a little
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive threshold to seperate the foreground from the
        # background
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

        # plot_images(gray, blur, thresh, 'gray', 'blurred', 'threshold')

        # Find all contours in the threshold image.
        contours, hire = cv2.findContours(thresh, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

        # Assumption: The biggest contour in the image is the sudoku grid.
        contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                          reverse=True)
        largest_contour = np.squeeze(contours[0])

        # grid_drawn = cv2.drawContours(original, contours, 0, (0, 255, 0), 3)
        # plot_single_image(grid_drawn, 'sudoku grid')

        # Create a mask image so that we ignore everything else outside the
        # grid and focus on the board
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        cv2.drawContours(mask, [largest_contour], 0, 0, 2)

        # Extract the region of interest using the mask generated above.
        out = np.zeros_like(gray)
        out[mask == 255] = gray[mask == 255]

        # Apply Gausian Blur to remove noise a little(to region of interest)
        blur_roi = cv2.GaussianBlur(out, (5, 5), 0)

        # Apply adaptive threshold.
        thresh_roi = cv2.adaptiveThreshold(blur_roi, 255, 1, 1, 11, 2)

        # plot_images(mask, out, thresh_roi, 'mask', 'roi', 'threshold roi')
        contours, _ = cv2.findContours(thresh_roi, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours in the region of interest with an area greater
        # than 800.
        c = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 800:
                cv2.drawContours(image, contours, c, (0, 255, 0), 3)
            c += 1

        plot_original_final(original, image)
