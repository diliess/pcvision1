import cv2 
import numpy as np

def over(pv):
    if(pv>255):
        return 255
    elif(pv<0):
        return 0
    else:
        return pv

def gaussian(img:np.ndarray):
    img_out=img.copy()
    kernel=np.array([[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]])
    row,col,channel=img.shape
    for i in range(channel):
        everyc=img[:,:,i]
        for j in range(row-2):
            for k in range(col-2):
                center=j+1,k+1
                p_point=img[j:j+3,k:k+3,i]

                p_out=kernel*p_point
                img_out[center[0],center[1]]=p_out.sum()
    return img_out
        
    
    

img=cv2.imread('data/noise3.png')
# img_test=np.array([[1,1,1,1],[2,2,2,3],[5,4,3,1]]).reshape(3,4,1)
outimg=cv2.GaussianBlur(img,(3,3),0,0)
print(outimg.shape)
cv2.imwrite("data/hand_write_gauss1.jpg",outimg[:,:,0])
cv2.imwrite("data/hand_write_gauss2.jpg",outimg[:,:,1])
cv2.imwrite("data/hand_write_gauss3.jpg",outimg[:,:,2])
cv2.imshow("origin",outimg) 

cv2.waitKey(0)
cv2.destroyAllWindows()


