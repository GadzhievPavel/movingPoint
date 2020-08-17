import cv2
import numpy as np

img1 = cv2.imread('/home/pavel/mai/res/RI1_0.jpg',0)

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 3 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

p0 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)
mask = np.zeros_like(img1)

for i in range(5):
        img2 = cv2.imread('/home/pavel/mai/res/RI1_'+str(i)+'.jpg',0)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)
        good_new = p1[st==1]
        good_old = p0[st==1]

        print("Create vectors")
        for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                print((a-c,b-d))
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                img2 = cv2.circle(img2,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(img2,mask)

        cv2.imshow('frame',img)
        cv2.imwrite("save.jpg",img)
        img1 = img2
        p0 = good_new.reshape(-1,1,2)
