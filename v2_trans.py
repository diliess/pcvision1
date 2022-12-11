import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('data/noise3.png',0) #0表示灰度图
img_float32 = np.float32(img)#转换格式
#傅里叶变化
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
#将图像频谱中的零频率分量会被移到频域图像的中心位置
dft_shift = np.fft.fftshift(dft)
#得到灰度图能表示的形式
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
