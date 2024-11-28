"""
该程序为机器视觉课程形状检测作业 Task1, 主要功能为基于Canny边界检测算法进行车道线检测。
作者: 曹翔宇
日期: 2024年4月25日
操作系统: Windows 11
环境: Python 3.9
依赖包: opencv-python, numpy
"""


import cv2
import numpy as np


# 非极大值抑制，细化边界宽度
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    # 新建一个相同大小梯度图保存抑制后的结果
    height, width = gradient_magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.uint8)

    # 将梯度方向由弧度转为角度并统一到 0~180°
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    # 逐像素进行非极大值抑制操作
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q, r = 255, 255
            # 按角度大小近似到0、45、90、135四个角度，选择相邻像素
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j+1]
                r = gradient_magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j + 1]
                r = gradient_magnitude[i - 1, j - 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i+1, j]
                r = gradient_magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j + 1]
                r = gradient_magnitude[i + 1, j - 1]

            # 对比相邻像素梯度值，若当前像素梯度值最大则保留，若不是最大值则将梯度置0
            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed


# 双阈值过滤与搜索连接
def double_threshold(image, low_thresh_ratio=0.95, high_thresh_ratio=0.3):
    # 按比例因子选择上下边界阈值
    max_value = image.max()
    high_thresh = max_value * high_thresh_ratio
    low_thresh = high_thresh * low_thresh_ratio
    print("pixel_value_max: ", max_value)
    print("high_thresh: ", high_thresh)
    print("low_thresh: ", low_thresh)

    # 逐像素进行分类，高于上界的为strong pixel，在上下界之间的为weak pixel，低于下界的为outlier
    strong_i, strong_j = np.where(image >= high_thresh)
    zeros_i, zeros_j = np.where(image < low_thresh)
    weak_i, weak_j = np.where((image < high_thresh) & (image >= low_thresh))
    print('find strong pixels: ', len(strong_i))
    print('find weak pixels: ', len(weak_i))
    print('find outliers: ', len(zeros_i))

    # strong pixel和weak pixel保留原梯度值，outlier梯度值置为0
    image[zeros_i, zeros_j] = 0

    # 搜索连接，对于weak pixel，如果与strong pixel相邻则保留，如果不相邻则置为0
    for n in range(len(weak_i)):
        i, j = weak_i[n], weak_j[n]
        if (image[i + 1, j - 1] >= high_thresh) or (image[i + 1, j] >= high_thresh) or (
                image[i + 1, j + 1] >= high_thresh) or (image[i, j - 1] >= high_thresh) or (
                image[i, j + 1] >= high_thresh) or (image[i - 1, j - 1] >= high_thresh) or (
                image[i - 1, j] >= high_thresh) or (image[i - 1, j + 1] >= high_thresh):
            continue
        else:
            image[i, j] = 0

    return image


def canny_edge_detection(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 步骤一 高斯模糊以降低噪声
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)

    # 步骤二 Sobel滤波计算梯度
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=1)

    # 计算原始图像梯度幅值图并保存
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_direction = np.arctan2(grad_y, grad_x)
    cv2.imwrite('task1_dst/original_gradient.png', grad_magnitude)

    # 步骤三 非极大值抑制处理梯度幅值图
    suppressed = non_maximum_suppression(grad_magnitude, grad_direction)
    cv2.imwrite('task1_dst/suppressed_gradient.png', suppressed)

    # 步骤四 双阈值过滤与搜索连接
    filtered_gradient = double_threshold(suppressed, low_thresh_ratio=0.8, high_thresh_ratio=0.35)
    cv2.imwrite('task1_dst/filtered_gradient.png', filtered_gradient)

    # 对结果进行二值化处理
    ret, binary_gradient = cv2.threshold(filtered_gradient, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite('task1_dst/binary_gradient.png', binary_gradient)

    return


def main():
    # 输入图像路径
    image_path = "src/lanes.png"

    # 读取图片
    image = cv2.imread(image_path)

    # Canny边缘检测
    canny_edge_detection(image)


if __name__ == "__main__":
    main()
