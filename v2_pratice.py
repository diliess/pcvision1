import cv2
import numpy as np

img = cv2.imread("data/A.png",cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)#二维傅里叶变换
f_shift = np.fft.fftshift(f)

magnitude_spectrum = 20*np.log(np.abs(f_shift))
magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)
img_and_spec = np.concatenate((img,magnitude_spectrum),axis=1)

cv2.imshow("",img_and_spec)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(magnitude_spectrum)