import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt



def preprocess_image(img):
    # Convert the image to gray-scale
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth it
    blured_gray = cv2.GaussianBlur(gray_scale, (11, 11), 0)


    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(blured_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

    # Makes the boarders white
    cv2.bitwise_not(thresh, thresh)

    # Dilate to connect back the disconnected parts
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)

    dilated = cv2.dilate(thresh, kernel)



    # plot_images(img, dilated)
    return dilated


def plot_images(img, img_final):
    plt.subplot(121), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_final, cmap='gray'), plt.title('final')
    plt.xticks([]), plt.yticks([])
    plt.show()


def find_biggest_blob(img):

    count = 0
    max = -1

    for i in range((img.size)):
        pass


def main():
    img = cv2.imread('playground/image1019.jpg')
    original = img.copy()

    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    preprocessed_image = preprocess_image(img)
    blob = find_biggest_blob(preprocessed_image)


    # plot_images(original, img)


if __name__ == '__main__':
    main()
