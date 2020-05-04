import fnmatch
import os
import cv2
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

    print(len(jpg_files))
    return jpg_files


def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def preprocess_image(img):
    # Convert the image to gray-scale
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth it
    # blured_gray = cv2.GaussianBlur(gray_scale, (3, 3), 0)
    blured_gray = cv2.GaussianBlur(gray_scale, (11, 11), 0)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(blured_gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 5, 2)

    # Makes the boarders white
    cv2.bitwise_not(thresh, thresh)

    # Dilate with a plus shaped kernel to connect back the disconnected parts
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)

    dilated = cv2.dilate(thresh, kernel)

    # plot_images(thresh, dilated)
    return dilated


def plot_images(img, img_final):
    plt.subplot(121), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_final, cmap='gray'), plt.title('final')
    plt.show()


def get_corners(img):
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
    img = input_image.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in corners:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)

    return img


def crop_sudoku_grid(img, corners):
    # Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = corners[0], corners[
        1], corners[2], corners[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left],
                   dtype='float32')

    # Get the longest side in the rectangle
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    # Describe a square with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    return cv2.warpPerspective(img, m, (int(side), int(side)))

    # roi = img[corners[0][1]:corners[2][1], corners[0][0]:corners[2][0]]
    #
    # return roi


def extract_rectangles(img, image):
    contours, hire = cv2.findContours(img, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000 / 2:
            cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    # plot_images(original, img, original)
    return image


if __name__ == '__main__':
    # images = ['playground/image1072.jpg', 'playground/image1024.jpg',
    #           'playground/image31.jpg']

    images = get_list_of_files('images')

    for image in images:
        img = cv2.imread(image)
        original = img.copy()

        # Preprocess Image
        preprocessed_image = preprocess_image(img)

        # Extract corners of the largest polynomial
        corners = get_corners(preprocessed_image)

        # corners_drawn = draw_corners(original, corners)
        cropped = crop_sudoku_grid(img, corners)

        plot_images(original, cropped)
