import numpy as np

img = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])
out = np.zeros((4,4,3))
for i in range(3):
    out[:,:,i] = img
thr =np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])

out[thr>2]=[100,0,0]

print(out)