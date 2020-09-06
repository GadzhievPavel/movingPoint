import cv2
import numpy as np
img1 = cv2.imread('/home/pavel/mai/res/i1.jpeg',0)

f = 2.4

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.4,
                       minDistance = 5,
                       blockSize = 15 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

p0 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)
mask = np.zeros_like(img1)

img2 = cv2.imread('/home/pavel/mai/res/i2.jpeg',0)
p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)
good_new = p1[st==1]
good_old = p0[st==1]

print("Create vectors")
x = list()
y = list()
for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #print((a-c,b-d))
        x.append((a-c))
        y.append((b-d))
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        img2 = cv2.circle(img2,(a,b),5,color[i].tolist(),-1)
img = cv2.add(img2,mask)
print(np.median(x),np.median(y))
###koeff
kX=0.3/np.median(x)
kY=0.3/np.median(y)

cv2.imshow('frame',img)
cv2.imwrite("save.jpg",img)
img1 = img2
p0 = good_new.reshape(-1,1,2)
