import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("data/face.png")

out1 = cv.Canny(image=img,threshold1=80,threshold2=100)
out2 = cv.Canny(image=img,threshold1=10,threshold2=200)

plt.subplot(2,2,1)
plt.imshow(out1)

plt.subplot(2,2,2)
plt.imshow(out2)

plt.subplot(2,2,3)
plt.imshow(img)
plt.show()

