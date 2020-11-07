'''
鲁棒性好
'''
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss


#  求两点的距离
def p2p_distance(p1, p2):
    d = math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))
    return d

#  求直线交点
def l2l_crossPoints(line1, line2):  # 计算交点函数
    #是否存在交点
    a,b=line1
    a.extend(b)
    line1=a
    a, b = line2
    a.extend(b)
    line2=a

    point_is_exist=False
    x=0
    y=0
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist=True
    elif k2 is None:
        x=x3
        y=k1*x3+b1
    elif not k2==k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist=True

    return point_is_exist,[x, y]#---用于输出查看详情

#  点到直线的距离
def p2l_distance(p, l):
    pointX = p[0]
    pointY = p[1]
    lineX1 = l[0]
    lineY1 = l[1]
    lineX2 = l[2]
    lineY2 = l[3]

    a = lineY2 - lineY1
    b = lineX1 - lineX2
    c = lineX2 * lineY1 - lineX1 * lineY2
    dis = (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))
    return dis

# 求众数
def seeking_modeNumber(nums):#-----输入list
    #统计每个数出现的次数
    counts = np.bincount(nums)#输出索引的个数,从零开始
    # 返回出现次数最多的数
    mode=0
    if len(counts):
        m = np.argmax(counts)#索引值对应目标值，索引的个数对应目标的频数
        mode=m  #None会直接返回，为避免None的返回，要通过赋值解决。None便是空，不被赋值
    return mode

def fitting_1center(image):
    # #1.寻找轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if image is None:
        print('拟合输入为空')
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img_h, img_w, img_c = image.shape
    image_center = (img_w / 2, img_h / 2)

    # 1.绘制所有的轮廓
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    # cv2.imshow('contours drawing',image)
    # cv2.waitKey()

    # 2.排除最大轮廓
    # print(contours)
    lines = []   #********************************************求刻度线
    for contour in contours:
        # 计算旋转矩形
        rect = cv2.minAreaRect(contour)  # 返回：中心(x,y)，(宽度，高度)，旋转角度

        if rect[1][0] != 0 or rect[1][1] != 0:
            rect_w = max(rect[1])
            rect_h = min(rect[1])

        rect_center = rect[0]  # -范围在[-90,0],大小为逆时针旋转到X轴的角度大小
        rect_angle = rect[2]
        box = cv2.boxPoints(rect)  # 返回：四个点
        box = np.int0(box)  # 化为整数

        # 计算轮廓的面积和周长
        # contour_area = cv2.contourArea(contour)  #计算contour 的面积
        # contour_perimeter = cv2.arcLength(contour, True)  #计算contour的周长

        # 绘制旋转矩形,拟合圆心
        # print(rect_w)
        # print(rect_h)
        if rect_h < 0.5 * img_h and rect_w < 0.5 * img_w:
            #           绘制旋转矩形
            cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
            p1 = box[0]
            p2 = box[1]
            p3 = box[2]
            p4 = box[3]

            p1 = [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]
            p2 = [int((p2[0] + p3[0]) / 2), int((p2[1] + p3[1]) / 2)]
            p3 = [int((p3[0] + p4[0]) / 2), int((p3[1] + p4[1]) / 2)]
            p4 = [int((p4[0] + p1[0]) / 2), int((p4[1] + p1[1]) / 2)]

            # print(p1)
            # print(p2)
            # print(p3)
            # print(p4)

            d1 = p2p_distance(p1, p3)
            d2 = p2p_distance(p2, p4)
            #           绘制刻度线
            if d1 > d2:
                cv2.line(image, tuple(p1), tuple(p3), (0, 0, 255), 3)
                line = []
                line.append(p1)
                line.append(p3)
                lines.append(line)

            else:
                cv2.line(image, tuple(p2), tuple(p4), (0, 0, 255), 3)
                line = []
                line.append(p2)
                line.append(p4)
                lines.append(line)
    # print(lines)


    # 求两条直线的交点：
    points = []
    for i in range(len(lines)):
        line1 = lines[0]
        for j in range(i, len(lines)):
            line2 = lines[j]
            point = l2l_crossPoints(line1, line2)
            points.append(point)

    # 绘制散点图：
    # print(points)
    points_cross=[]
    for i in range(len(points)):
        point=points[i]
        # print(point[0])
        # print(point[1])
        if point[0]==True:
            points_cross.append(point[1])

    #Points 转int：
    array=np.array(points_cross)
    array=np.around(array)
    array=array.astype(int)
    array=array.tolist()
    cross_points=array
    # print(len(cross_points))

    #绘制拟合点:
    for i in range(len(cross_points)):
        cv2.circle(image,tuple(cross_points[i]),1,(255,0,0),1)
    #求取交点众数：
    x=[]
    y=[]
    for p in cross_points:
        x.append(p[0])
        y.append(p[1])
    # print(x)
    # print(y)
    for i in x[:]:
        if i<0:
            x.remove(i)
    for j in y[:]:
        if j<0:
            y.remove(j)

    x=seeking_modeNumber(x)
    y=seeking_modeNumber(y)
    # x=ss.stats.mode(x)[0][0]
    # y=ss.stats.mode(y)[0][0]
    # print(x)
    # print(y)
    cv2.circle(image,(x,y),3,(0,0,255),3)
    center=[x,y]

    # print(points_cross)
    # cv2.imshow("inmg", image)
    # cv2.waitKey(500)
    return image  #-------出图的时候用(可视化)
    # cv2.imshow('image',image)
    # cv2.waitKey()
    # return center  #----输出结果

## 循环输出点进行center拟合--批量的拟合*************输入二值灰度图,输出图（不是圆心哦）
def centerFitting():
    image_dir='./scales'
    save_dir='./out_scales'
    image_name=os.listdir(image_dir)
    images=list()
    for pic_name in image_name:
        #读图：
        image_path=image_dir+'/'+pic_name
        image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        # image=cv2.bitwise_not(image)
        cv2.imshow('img',image)
        cv2.waitKey(1000)
        # #拟合圆心：
        image=fitting_1center(image)

        # images.append(image)
        #保存结果：
        out_path = save_dir + '/' + pic_name
        cv2.imwrite(out_path, image)  #注意==fitting_1center的return要返回image
        cv2.imshow('img', image)
        cv2.waitKey(300)

if __name__ == '__main__':
    centerFitting()

