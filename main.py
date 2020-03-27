import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


def preprocess_image(img):
    # Convert the image to gray-scale
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth it
    blured_gray = cv2.GaussianBlur(gray_scale, (5, 5), 0)

    # plot_images(img, blured_gray)
    return blured_gray


def plot_images(img, img_final):
    plt.subplot(121), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_final, cmap='gray'), plt.title('final')
    plt.xticks([]), plt.yticks([])
    plt.show()


# def get_corners_of_largest_poly(edges):
#     contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True) #Sort by area, descending
#     polygon = contours[0]  # get largest contour
#
#     bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
#     top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
#     bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
#     top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
#
#     return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def extract_lines(img):
    horizontal_lines = []
    vertical_lines = []

    intersetction_points = []

    edges = cv2.Canny(img, 30, 60, apertureSize=3)

    hough_lines = cv2.HoughLines(edges, 2, np.pi/180, 300, 0, 0)
    hough_lines = sorted(hough_lines, key=lambda line:line[0][0])

    pos_horizontal = 0
    pos_vertical = 0
    for line in hough_lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 10000 * (-b))
            y1 = int(y0 + 10000 * a)
            x2 = int(x0 - 10000 * (-b))
            y2 = int(y0 - 10000 * a)

            # Vertical if close to 180 or 0 degree.
            # if theta < np.pi/20 or theta > np.pi - np.pi/20:
            #     vertical_lines.append(Line(x1, y1, x2, y2))
                # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Horizontal if close to 90 degree
            # elif abs(theta - np.pi/2) < np.pi/20:
            #     horizontal_lines.append(Line(x1, y1, x2, y2))
                # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


            if b > 0.5:
                # needs to be a vertical line
                if rho - pos_horizontal > 10:
                    vertical_lines.append(Line(x1, y1, x2, y2))
                    pos_horizontal = rho
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                # horizontal line
                if rho - pos_vertical > 10:
                    horizontal_lines.append(Line(x1, y1, x2, y2))
                    pos_vertical = rho
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # for h_line in horizontal_lines:
    #     for v_line in vertical_lines:
    #         pass

    return img


def main():
    img = cv2.imread('playground/image1024.jpg')

    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    preprocessed_image = preprocess_image(img)

    lines = extract_lines(preprocessed_image)

    # perform edge detection
    # edges = cv2.Canny(preprocessed_image, 30, 60, apertureSize=3)

    # extract corner points from edges
    # corners = get_corners_of_largest_poly(edges)

    # detect lines in the image using hough lines technique
    # lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # img_final = img.copy()
    # marked_corners = draw_corners(img_final, corners)

    plot_images(img, lines)


if __name__ == '__main__':
    main()
