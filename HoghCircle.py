'''
都能检测出来、就是误差太大
'''
import cv2
import numpy as np
import math
import cv2
import os

#基于圆度的园识别
def circlerecognation(img):
        # gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        gray=img
        minDists = [100,125,150]
        imgcopy = [img.copy(),img.copy(),img.copy()]
        for minDist,imgcopy in zip(minDists,imgcopy):
            circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1,minDist=minDist,param1=100,param2=30,minRadius=20,maxRadius=300)
            for circle in circles[0]:
                x=int(circle[0])
                y=int(circle[1])
                r=int(circle[2])
                img=cv2.circle(imgcopy,(x,y),r,(0,0,255),2)
            cv2.imshow('circle_img_'+str(minDist),img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
def distance(w,h,p):

    d=math.sqrt(pow(p[0]-w/2,2)+pow(p[1]-h/2,2))
    return d

def houghCircle(image):
    # dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=5) # 高斯双边滤波(慢)
    dst = cv2.pyrMeanShiftFiltering(image, 10, 100)                           # 均值偏移滤波（稍微快）
    dst = cv2.cvtColor(dst, cv2.COLOR_BGRA2GRAY)

    # cv2.imshow("adapt_image", dst)
    circle = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=50, maxRadius=300)
    if not circle is None:
        circle = np.uint16(np.around(circle))
        print(circle)
        for i in circle[0, :]:
            p=(i[0],i[1])
            h, w, c = image.shape
            d=distance(w,h,p)
            if d<100:

                cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
                cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
                # cv2.circle(image,(int(w/2),int(h/2)),2,(255,0,0),3)
    return image

if __name__ == "__main__":

    dir='./images'
    files=os.listdir(dir)
    for file in files:
        img_path=dir+'/'+file
        img = cv2.imread(img_path)
        # src = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        # image=circlerecognation(src)#--01
        image=houghCircle(img)#-----02

        cv2.imshow("circle", img)
        cv2.waitKey(100)
        # cv2.destroyAllWindows()
        # cv2.imwrite('./4.png',image)