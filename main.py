import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt


def preprocess_image(img):
    # Copy the image
    outer_box = img.copy()

    # Convert the image to gray-scale
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth it
    blur = cv2.GaussianBlur(gray_scale, (11, 11), 0)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2, dst=outer_box)

    cv2.bitwise_not(outer_box, outer_box)

    kernel = np.ones((5, 5), np.uint8)

    # Used to make the volume of the grid larger.
    dilation = cv2.dilate(outer_box, kernel, iterations=1)


    # return outer_box
    return dilation


def plot_images(img, img_final):
    plt.subplot(121), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(gray_scale, cmap='gray'), plt.title('final')
    plt.subplot(122), plt.imshow(img_final, cmap='gray'), plt.title('final')
    plt.xticks([]), plt.yticks([])
    plt.show()


def draw_lines(img_final, lines):
    # define position of horizontal line and vertical line
    # pos_horizontal = 0
    # pos_vertical = 0

    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 10000 * (-b))
            y1 = int(y0 + 10000 * (a))
            x2 = int(x0 - 10000 * (-b))
            y2 = int(y0 - 10000 * (a))

            cv2.line(img_final, (x1, y1), (x2, y2), (0, 0, 255), 3)


def get_corners_of_largest_poly(edges):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True) #Sort by area, descending
    polygon = contours[0]  # get largest contour

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def draw_corners(input_image, corners, radius=5, colour=(0, 0, 255)):
    img = input_image.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in corners:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)

    return img

def main():
    img = cv2.imread('playground/image1019.jpg')

    preprocessed_image = preprocess_image(img)

    # perform edge detection
    edges = cv2.Canny(preprocessed_image, 30, 60, apertureSize=3)

    # extract corner points from edges
    corners = get_corners_of_largest_poly(edges)

    # detect lines in the image using hough lines technique
    # lines = cv2.HoughLines(edges, 2, np.pi/180, 300, 0)

    img_final = img.copy()
    marked_corners = draw_corners(img_final, corners)
    # draw_lines(img_final, lines)
    plot_images(img, marked_corners)


if __name__ == '__main__':
    main()
