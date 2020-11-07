'''
光照敏感、噪声敏感
'''

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def compute_roundness(contour):
    # contours, hierarchy = cv2.findContours(np.array(label_image, dtype=np.uint8), cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_NONE)
    a = cv2.contourArea(contour) * 4 * math.pi
    b = math.pow(cv2.arcLength(contours[0], True), 2)
    if b == 0:
        return 0
    return a / b


def alpha_fill(image, contour):
    zeros = np.zeros(image.shape, dtype=np.uint8)

    cv2.fillPoly(zeros, contour, color=(0, 255, 0))  # ------------输入点集合用
    # cv2.fillConvexPoly(zeros, contour, (0, 255, 0))
    cv2.imwrite('./mask.png', zeros)

    mask = cv2.imread('mask.png')
    try:
        mask_img = mask + image

        alpha1 = 1  # alpha 为第一张图片的透明度==原始图片
        alpha2 = 0.8  # beta 为第二张图片的透明度

        gamma = 0  # 曝光度
        img = cv2.addWeighted(image, alpha1, mask, alpha2, gamma)
    except:
        print('异常')
    return img


if __name__ == '__main__':
    #   1.读图
    img = cv2.imread('./001.jpg')
    img2 = cv2.imread('./001.jpg', cv2.IMREAD_GRAYSCALE)  # 透明俺膜使用的底照

    #   2.灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, 200, 250)

    #   3.轮廓寻找
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    # cv2.imshow('contours',img)
    # cv2.waitKey(10)

    # #面积法
    # area = []
    # for i in range(len(contours)):
    #     area.append(cv2.contourArea(contours[i]))
    # max_idx = np.argmax(area)  #---返回最大值索引
    #
    #   4.圆度法
    rounds = []
    for i in range(len(contours)):
        r = compute_roundness(contours[i])
        rounds.append(r)
    max_idx = np.argmax(rounds)
    # print(rounds)

    #    5.填充方式：
    # cv2.fillConvexPoly(img, contours[max_idx], (0,255,0))#------填充convex
    # cv2.fillPoly(img,contours[max_idx],(0,255,0))  #----------填充任意多边形
    img = alpha_fill(img, [contours[max_idx]])  # -------------透明填充

    #   5. 椭圆拟合
    ellipse = cv2.fitEllipse(contours[max_idx])
    cv2.ellipse(img, ellipse, (0, 0, 255), 2, cv2.LINE_AA)
    x, y = ellipse[0]
    cv2.circle(img, (np.int(x), np.int(y)), 4, (0, 0, 255), -1, 8, 0)
    center = (np.int(x), np.int(y))
    c_str = str(center)
    # cv2.putText(img,c_str,(center[0],center[1]+30,),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(0,0,255),thickness=2)

    #   6.  显示
    cv2.imshow('result', img)
    cv2.waitKey()



