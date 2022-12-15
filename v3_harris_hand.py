import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("data/conner1.webp")
gary_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# print(gary_img.shape)

dx = cv.Sobel(gary_img,cv.CV_32F,1,0,ksize=3)
dy = cv.Sobel(gary_img,cv.CV_32F,0,1,ksize=3)

out_img = np.zeros([img.shape[0],img.shape[1]],dtype=np.float32)
computer_img = np.zeros([img.shape[0],img.shape[1],img.shape[2]],dtype=np.float32)

xlen = gary_img.shape[0]
ylen = gary_img.shape[1]
# print(xlen,ylen)

for x in range(xlen):#计算矩阵储存dx dy
    for y in range(ylen):
        computer_img[x,y,0]=int(dx[x,y])*int(dx[x,y])
        computer_img[x,y,1]=int(dy[x,y])*int(dy[x,y])
        computer_img[x,y,2]=int(dx[x,y])*int(dy[x,y])
# computer_img = cv.boxFilter(computer_img,-1,(2,2),normalize=False)
for x in range(out_img.shape[0]):
    for y in range(out_img.shape[1]):
        out_img[x,y]=computer_img[x,y,0]*computer_img[x,y,1]-computer_img[x,y,2]**2-0.04*(computer_img[x,y,0]+computer_img[x,y,1])**2        #0.04是参数

out_img=abs(out_img)
out_img=out_img/out_img.max()*256

for x in range(out_img.shape[0]):
    for y in range(out_img.shape[1]):
        if(out_img[x,y]>200):
            img[x,y]=[255,0,0]


# print(out_img.max())
# plt.subplot(1,2,1)
plt.imshow(img)
# plt.imshow(out_img)   
plt.show()            
