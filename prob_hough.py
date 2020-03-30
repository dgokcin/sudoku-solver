import cv2
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

    # Dilate with a plus shaped kernel to connect back the disconnected parts
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


def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])

    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
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
    roi = img[corners[0][1]:corners[2][1], corners[0][0]:corners[2][0]]
    return roi


def draw_lines(img):
    # Use Canny edge detection and dilate the edges for better result.
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)

    minLineLength = 350
    maxLineGap = 20
    lines = cv2.HoughLinesP(dilation, 1, np.pi / 180, 50, minLineLength, maxLineGap)
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except TypeError:
        print("Could not detect any lines. ")

    return img


if __name__ == '__main__':
    img = cv2.imread('playground/image1072.jpg')
    original = img.copy()

    # Preprocess Image
    preprocessed_image = preprocess_image(img)

    # Extract corners of the largest polynomial
    corners = get_corners(preprocessed_image)

    # corners_drawn = draw_corners(original, corners)
    # cropped = crop_sudoku_grid(img, corners)

    # apply
    final = draw_lines(img)

    plot_images(original, final)

