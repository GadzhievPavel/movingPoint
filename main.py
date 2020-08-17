import cv2
import numpy as np 
import math
from numpy import median
from matplotlib import pyplot as plt 

def calcTeta(img,x1,y1,x2,y2):
    halfY = img.shape[0]/2
    halfX = img.shape[1]/2
    vec1 = math.sqrt((x1-halfX)*(x1-halfX)+(y1-halfY)*(y1-halfY))
    vec2 = math.sqrt((x2-halfX)*(x2-halfX)+(y2-halfY)*(y2-halfY))
    vec3 = math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
    alpha = math.acos(((vec1*vec1)+(vec2*vec2)-(vec3*vec3))/(2*vec1*vec2))
    return alpha

orb = cv2.ORB_create()

for i in range(5):
    print("KeyPoint"+str(i))
    img1 = cv2.imread('/home/pavel/mai/res/RI1_'+str(i)+'.jpg',0)
    img2 = cv2.imread('/home/pavel/mai/res/RI1_'+str(i+1)+'.jpg',0)
    halfWidth = img1.shape[1]/2
    halfHeight = img2.shape[0]/2
    orb.setMaxFeatures(20)
    orb.setEdgeThreshold(10)
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    coord = list()
    alphaList = list()
    for j in range(len(kp1)):
        if (j < len(kp2)):
            deltaX = kp1[j].pt[0] - kp2[j].pt[0]
            deltaY = kp1[j].pt[1] - kp2[j].pt[1]
            alphaList.append(calcTeta(img1,kp1[j].pt[0],kp1[j].pt[1],kp2[j].pt[0],kp2[j].pt[1]))
            print("alpha "+ str(calcTeta(img1,kp1[j].pt[0],kp1[j].pt[1],kp2[j].pt[0],kp2[j].pt[1])))
            coord.append((deltaX,deltaY))
        else:
            break 

    print( "median angle"+ str(median(alphaList)))
    print(coord)
    medianAngle = median(alphaList)    

    coordX=list()
    coordY=list()
    if medianAngle < 1:
        for c in coord:
            coordX.append(c[0])
            coordY.append(c[1])
    X = median(coordX)
    Y = median(coordY)
    print("X: "+str(X)+" Y: "+str(Y)+" angle: "+str(medianAngle))        
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], outImg=img1, flags=0)
    plt.imshow(img3)
    plt.text(0, -10, "X: "+str(X)+" Y: "+str(Y)+" angle: "+str(medianAngle), fontsize=15)
    plt.show()