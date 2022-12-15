import cv2
import numpy as np
import matplotlib.pyplot as plt
#读入图像并转化为float类型，用于传递给harris函数
filename = 'data/harris.jpeg'
 
img = cv2.imread(filename)
 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
gray_img = np.float32(gray_img)
 
#对图像执行harris
Harris_detector = cv2.cornerHarris(gray_img, 2, 3, 0.04)
 
#膨胀harris结果
# dst = cv2.dilate(Harris_detector, None)
 
# 设置阈值
thres = 0.01*Harris_detector.max()
 
img[Harris_detector > thres] = [255,0,0]
 
print(Harris_detector.shape)