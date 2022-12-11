import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('data/noise3.png',0) #0表示灰度图
img_float32 = np.float32(img)#转换格式
#傅里叶变化
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
#将图像频谱中的零频率分量会被移到频域图像的中心位置
dft_shift = np.fft.fftshift(dft)
rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2) # 中心位置
# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0
# IDFT傅里叶逆变化
fshift = dft_shift*mask#保留中间部分
f_ishift = np.fft.ifftshift(fshift)#将零频率移回原来位置
img_back = cv2.idft(f_ishift)# IDFT傅里叶逆变化
#得到灰度图能表示的形式
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()
