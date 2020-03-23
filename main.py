import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('playground/image18.jpg')
img_final = cv2.imread('playground/image18.jpg')

# Convert the image to gray-scale
gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# perform edge detection
edges = cv2.Canny(gray_scale, 50, 150, apertureSize=3)

# detect lines in the image using hough lines technique
lines = cv2.HoughLines(edges, 1, np.pi/180, 80, 200)

# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# iterate over the output lines and draw them
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

        cv2.line(img_final, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.subplot(121), plt.imshow(img), plt.title('original')
plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(gray_scale, cmap='gray'), plt.title('final')
plt.subplot(122), plt.imshow(img_final, cmap='gray'), plt.title('final')
plt.xticks([]), plt.yticks([])
plt.show()
