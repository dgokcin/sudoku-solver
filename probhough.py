import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_images(img, img_final):
    plt.subplot(121), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_final, cmap='gray'), plt.title('final')
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':

    gray = cv2.imread('playground/image1072.jpg')
    original = gray.copy()
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    cv2.imwrite('edges-50-150.jpg',edges)
    minLineLength = 500
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

    a,b,c = lines.shape
    for i in range(a):
        cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imwrite('houghlines5.jpg',gray)
    plot_images(original, gray)