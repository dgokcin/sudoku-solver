import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('playground/image18.jpg')
img_final = cv2.imread('playground/image18.jpg')

# Convert the image to gray-scale
gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# perform edge detection
edges = cv2.Canny(gray_scale, 75, 150)

# detect lines in the image using hough lines technique
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, np.array([]), maxLineGap=50)

# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img_final, (x1, y1), (x2, y2), (20, 220, 20), 3)

plt.subplot(121), plt.imshow(img), plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_final), plt.title('final')
plt.xticks([]), plt.yticks([])
plt.show()
