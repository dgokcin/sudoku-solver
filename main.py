import cv2
import numpy as np
from matplotlib import pyplot as plt


def magic_function(img):
    original = img.copy()

    # Smooth it
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert the image to gray-scale
    gray_scale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_scale, 30, 60, apertureSize=3)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 5, 2)

    # Makes the boarders white
    cv2.bitwise_not(thresh, thresh)


    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)

    dilated = cv2.dilate(thresh, kernel)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plot_images(original, dilated, img)
    # return original


def plot_images(img, img_cst, img_final):
    plt.subplot(131), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_cst, cmap='gray'), plt.title('before '
                                                                 'hough')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_final, cmap='gray'), plt.title('final')
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    images = ['playground/image1019.jpg', 'playground/image18.jpg',
              'playground/image25.jpg']
    for image in images:
        img = cv2.imread(image)

        magic_function(img)
