import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt


def preprocess_image(img):
    # Convert the image to gray-scale
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_scale


def plot_images(img, img_final):
    plt.subplot(121), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(gray_scale, cmap='gray'), plt.title('final')
    plt.subplot(122), plt.imshow(img_final, cmap='gray'), plt.title('final')
    plt.xticks([]), plt.yticks([])
    plt.show()


def draw_lines(img_final, lines):
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))


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
    img = cv2.imread('playground/image18.jpg')

    processed_image = preprocess_image(img)

    # perform edge detection
    edges = cv2.Canny(processed_image, 50, 150, apertureSize=3)

    # extract corner points from edges
    corners = get_corners_of_largest_poly(edges)

    # detect lines in the image using hough lines technique
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    img_final = img.copy()
    # marked_corners = draw_corners(img_final, corners)
    draw_lines(img_final, lines)
    plot_images(img, img_final)

if __name__ == '__main__':
    main()
