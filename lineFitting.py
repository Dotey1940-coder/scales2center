'''
pca具有降低噪声的性能，故表现固然优于最小二乘法、hough直线检测不适用于直线拟合，只适合直线检测
'''
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def read_pixels(image):
    # 1.访问图像：提取像素x,y
    h, w = image.shape
    x = list()
    y = list()
    for i in range(h):
        for j in range(w):
            # print(image[i, j])
            # pixel = image.item(i, j)
            if image[i, j] == 0:
                x.append(j)  # W为宽=====x
                y.append(i)  # H为高=======y

    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    # print(x.shape)
    # print(y.shape)
    data = np.vstack((x, y))
    print('输入的data')
    print(data.shape)
    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ls====01
def ls(data):
    data = data.T  # 转置
    print('pca的data')
    print(data.shape)
    # 绘制源点集合
    plt.plot(data[:, 0], data[:, 1], 'k,')
    plt.legend('data', shadow=True, )

    print(data.shape)
    N = np.size(data, 0)
    coeMatrix = np.vstack((data[:, 0], np.ones(N))).transpose()
    coeRhs = data[:, 1]
    A = np.dot(coeMatrix.transpose(), coeMatrix)
    f = np.dot(coeMatrix.transpose(), coeRhs)
    kb = np.linalg.solve(A, f)
    k = kb[0]
    b = kb[1]
    x_ls = np.linspace(1, 500)
    y_ls = x_ls * k + b

    plt.plot(x_ls, y_ls, 'b', linewidth=1)
    plt.legend('least square', shadow=True, )
    plt.axis('equal')
    plt.xlim(0, 620)
    plt.ylim(620, 0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~pca-==============02
def pca(data):
    data = data.T
    N = np.size(data, 0)
    print(N)
    dataHomo = data.copy()
    dataHomo[:, 0] -= np.sum(data[:, 0]) / N
    dataHomo[:, 1] -= np.sum(data[:, 1]) / N
    dataMatrix = np.dot(dataHomo.transpose(), dataHomo)
    u, s, vh = np.linalg.svd(dataMatrix, full_matrices=True)
    n = u[:, -1]
    k2 = -n[0] / n[1]
    b2 = np.sum(data[:, 1]) / N - k2 * np.sum(data[:, 0]) / N
    x_pca = np.linspace(1, 500)
    y_pca = x_pca * k2 + b2

    plt.plot(x_pca, y_pca, 'r', linewidth=1)
    plt.legend(('data', 'least square', 'pca'), shadow=True, )
    plt.axis('equal')
    plt.xlim(0, 620)
    plt.ylim(620, 0)

if __name__ == "__main__":
    image_dir = "./pointers"
    out_dir = './out_pointers'
    names = os.listdir(image_dir)
    for name in names:
        image_path = image_dir + '/' + name
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = 255 - img
        cv2.imshow('pic', img)
        cv2.waitKey(500)
        image = np.array(img)
        print(image.shape)
        fg = plt.figure(facecolor='k')
        plt.fill([0, 800], [800, 0], 'k')
        # 1.
        data = read_pixels(image)
        # 2.
        ls(data)  # ---------------------------------最小二乘法拟合直线
        # 3.
        pca(data)  # ---------------------------------pca拟合直线

        # 4. 保存点：
        # point=list(zip(x,y))
        # point=np.array(point)
        # np.savetxt('C:/Users/Dotey1940/Desktop/_1_.txt',point,'%.2f',delimiter=",")

        # 5.保存figure
        out_path = out_dir + '/' + name

        fig = plt.gcf()
        fig.set_facecolor('k')
        x = 0
        y = 0
        plt.fill(x, y, 'r')

        plt.savefig(out_path, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print('have done!!!1')
