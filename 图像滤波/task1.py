"""
该程序为机器视觉课程滤波器作业 Task1, 主要功能为基于 OpenCV 采用多个滤波器对图像进行处理。
作者: 曹翔宇
日期: 2024年4月7日
操作系统: Windows 11
环境: Python 3.9
依赖包: opencv-python, opencv-contrib-python, numpy
"""

import os

import cv2
import numpy as np


# 定义不同的滤波器
def apply_filters(image):
    # 3x3 Sobel x 方向
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)

    # 3x3 Sobel y 方向
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)

    # 导数滤波（幅值）
    kernel_x = np.array([[1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1], [0], [-1]], dtype=np.float32)
    derivative_x = cv2.filter2D(image, cv2.CV_64F, kernel_x, borderType=cv2.BORDER_REPLICATE)
    derivative_y = cv2.filter2D(image, cv2.CV_64F, kernel_y, borderType=cv2.BORDER_REPLICATE)

    # 3x3 均值滤波
    mean_blur = cv2.blur(image, (3, 3), borderType=cv2.BORDER_REPLICATE)

    # 3x3 中值滤波
    median_blur = cv2.medianBlur(image, 3)

    # 3x3 高斯滤波，sigmaX=0.8，sigmaY=0.8
    gaussian_blur = cv2.GaussianBlur(image, (3, 3), sigmaX=0.8, sigmaY=0.8, borderType=cv2.BORDER_REPLICATE)

    return sobel_x, sobel_y, derivative_x, derivative_y, mean_blur, median_blur, gaussian_blur


def main():
    # 输入图像路径
    input_image_paths = ["src/origin_image.png", "src/gaussian_noise.png", "src/pepper_noise.png"]

    # 输入图片与滤波器标签
    imgs = ['origin', 'gaussian', 'pepper']
    filters = ["Sobel_x", "Sobel_y", "Derivation_x", "Derivation_y", "Mean", "Median", "Gaussian"]

    for i, input_image_path in enumerate(input_image_paths):
        # 读取输入图像
        image = cv2.imread(input_image_path)

        # 应用滤波器得到输出结果
        filtered_images = apply_filters(image)

        # 保存结果，例 3x3_Sobel_pepper
        for j in range(len(filters)):
            cv2.imwrite(os.path.join('task1_dst', imgs[i], f"3x3_{filters[j]}_{imgs[i]}.png"), filtered_images[j])


if __name__ == "__main__":
    main()
