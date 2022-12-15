import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread("pcvision1/data/circle.png")

img_sobely = cv.Sobel(img,-1,0,1,ksize=3)
img_sobelx = cv.Sobel(img,-1,1,0,ksize=3)
img_sobely_abs = cv.convertScaleAbs(img_sobely)
img_sobelx_abs = cv.convertScaleAbs(img_sobelx)

out = cv.addWeighted(img_sobelx_abs,0.5,img_sobely_abs,0.5,0)
plt.subplot(2,3,1)
plt.imshow(img_sobelx)
plt.subplot(2,3,2)
plt.imshow(img_sobely)
plt.subplot(2,3,3)
plt.imshow(img_sobelx_abs)
plt.subplot(2,3,4)
plt.imshow(img_sobely_abs)
plt.subplot(2,3,5)
plt.imshow(out)
plt.show()
print(img_sobely_abs==img_sobelx)