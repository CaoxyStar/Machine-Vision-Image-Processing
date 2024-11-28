"""
该程序为机器视觉课程形状检测作业 Task2, 主要功能为基于Canny边界检测与霍夫圆算法进行车轮检测。
作者: 曹翔宇
日期: 2024年4月26日
操作系统: Windows 11
环境： Python 3.9
依赖包: opencv-python, numpy, tqdm, math
"""


import cv2
import numpy as np
from tqdm import tqdm
import math


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

    # 逐像素进行分类，高于上界的为strong pixel，在上下界之间的为weak pixel，低于下界的为outlier
    strong_i, strong_j = np.where(image >= high_thresh)
    zeros_i, zeros_j = np.where(image < low_thresh)
    weak_i, weak_j = np.where((image < high_thresh) & (image >= low_thresh))

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


# Canny边缘检测
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

    # 步骤三 非极大值抑制处理梯度幅值图
    suppressed = non_maximum_suppression(grad_magnitude, grad_direction)

    # 步骤四 双阈值处理与搜索连接
    filtered_gradient = double_threshold(suppressed, low_thresh_ratio=0.8, high_thresh_ratio=0.35)

    # 对结果进行二值化处理
    ret, binary_gradient = cv2.threshold(filtered_gradient, 0, 255, cv2.THRESH_BINARY)

    return binary_gradient


# 霍夫圆求解
def hough_circle(edges, min_radius=1, max_radius=200, min_dist=20):
    # 霍夫圆变换参数
    num_rows, num_cols = edges.shape
    max_radius = min(max_radius, min(num_rows, num_cols) // 2)

    # 构建累加器
    accumulator = np.zeros((num_rows, num_cols, max_radius), dtype=np.uint8)

    # 逐像素处理
    for y in tqdm(range(num_rows)):
        for x in range(num_cols):
            # 只对有梯度的像素操作
            if edges[y, x] > 0:
                for r in range(min_radius, max_radius):
                    for angle in range(0, 360):
                        a = x - int(r * np.cos(np.radians(angle)))
                        b = y - int(r * np.sin(np.radians(angle)))
                        if 0 <= a < num_cols and 0 <= b < num_rows:
                            accumulator[b, a, r] += 1

    # 找到累加器中的圆
    detected_circles = []
    for y in range(num_rows):
        for x in range(num_cols):
            for r in range(min_radius, max_radius):
                if accumulator[y, x, r] >= min_dist:
                    detected_circles.append(((x, y), r, accumulator[y, x, r]))

    # 对投票数进行可视化，为方便在二维图像上显示，对每个像素不同半径下的投票数取均值进行可视化
    vote_vis = np.average(accumulator, axis=2)

    # 对检测到的圆进行NMS处理，过滤掉重复检测到的圆
    filtered_circles = circle_nms(detected_circles)

    return filtered_circles, vote_vis


# 霍夫圆形过滤条件
def circle_filter(circle1, circle2, min_diff=10):
    (x1, y1), r1, _ = circle1
    (x2, y2), r2, _ = circle2
    # 如果两圆的圆心距小于半径的一半且半径差小于min_diff则视为重复圆，需要滤掉
    if math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)) < r1 / 2 and r1 - r1 < min_diff:
        return True
    else:
        return False


# 霍夫圆NMS，用于霍夫圆的去重
def circle_nms(circles):
    # 对检测到的圆按照投票数进行排序
    sorted_circles = sorted(circles, key=lambda x: x[2], reverse=True)
    # keep存保留的圆，retain用来更新每一次循环后剩下的圆
    keep, retain = [], []
    while sorted_circles:
        keep.append(sorted_circles[0])
        # 如果当前圆需要滤掉则跳到下一个圆，如果没有滤掉则加入retain列表
        for circle in sorted_circles[1:]:
            if circle_filter(sorted_circles[0], circle):
                continue
            else:
                retain.append(circle)
        sorted_circles = retain
        retain = []
    return keep


def main():
    # 输入图像路径
    image_path = "src/wheel.png"

    # 读取图片
    image = cv2.imread(image_path)

    # Canny边缘检测
    edges = canny_edge_detection(image)
    cv2.imwrite('task2_dst/detected_edges.png', edges)

    # 霍夫圆检测
    circles, vote_vis = hough_circle(edges, min_radius=10, max_radius=20, min_dist=200)

    # 保存投票结果，对每个像素上各半径的投票数取均值显示
    cv2.imwrite('task2_dst/vote_vis.png', vote_vis)

    # 打印检测到的车轮坐标、半径与投票数
    print("Detected circles: ", circles)

    # 在原图绘制霍夫圆
    color = (0, 0, 255)  # (B, G, R)
    thickness = 2
    for item in circles:
        center, radius, _ = item
        cv2.circle(image, center, radius, color, thickness)

    # 保存绘制结果
    cv2.imwrite('task2_dst/img_circle.png', image)


if __name__ == "__main__":
    main()
