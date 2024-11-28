"""
该程序为机器视觉课程频域作业 Task1, 实现了对于灰度图的腐蚀、膨胀、开运算、闭运算、顶帽运算和黑帽运算。
作者: 曹翔宇
日期: 2024年4月9日
操作系统: Windows 11
环境: Python 3.9
依赖包: opencv-python, numpy
"""

import os

import cv2
import numpy as np


# 腐蚀运算
def erosion(image, kernel_size=(3, 3)):
    # 获取图像的高度和宽度
    height, width = image.shape

    # 获取kernel的高度和宽度
    k_height, k_width = kernel_size

    # 计算kernel的半径
    k_radius = k_height // 2

    # 创建一个空白图像用于存储结果
    result_image = np.zeros((height, width), dtype=np.uint8)

    # padding
    padded_image = np.full((height + 2 * k_radius, width + 2 * k_radius), 255, dtype=np.uint8)
    padded_image[k_radius:height + 1, k_radius:width + 1] = image

    # 遍历图像的每个像素
    for i in range(k_radius, height):
        for j in range(k_radius, width):
            # 获取kernel覆盖区域的像素值
            roi = image[i - k_radius:i + k_radius + 1, j - k_radius:j + k_radius + 1]

            # 计算kernel区域内的最小像素值
            min_value = np.min(roi)

            # 将结果图像的对应位置设置为最小像素值
            result_image[i, j] = min_value

    return result_image


# 膨胀运算
def dilation(image, kernel_size=(3, 3)):
    # 获取图像的高度和宽度
    height, width = image.shape

    # 获取kernel的高度和宽度
    k_height, k_width = kernel_size

    # 计算kernel的半径
    k_radius = k_height // 2

    # 创建一个空白图像用于存储结果
    result_image = np.zeros((height, width), dtype=np.uint8)

    # padding
    padded_image = np.full((height + 2 * k_radius, width + 2 * k_radius), 0, dtype=np.uint8)
    padded_image[k_radius:height + 1, k_radius:width + 1] = image

    # 遍历图像的每个像素
    for i in range(k_radius, height):
        for j in range(k_radius, width):
            # 获取kernel覆盖区域的像素值
            roi = image[i - k_radius:i + k_radius + 1, j - k_radius:j + k_radius + 1]

            # 计算kernel区域内的最大像素值
            max_value = np.max(roi)

            # 将结果图像的对应位置设置为最大像素值
            result_image[i, j] = max_value

    return result_image


# 开运算
def opening(image, kernel_size=(3, 3)):
    return dilation(erosion(image, kernel_size), kernel_size)


# 闭运算
def closing(image, kernel_size=(3, 3)):
    return erosion(dilation(image, kernel_size), kernel_size)


# 顶帽运算
def tophat(image, kernel_size=(3, 3)):
    return image - opening(image, kernel_size)


# 黑帽运算
def blackhat(image, kernel_size=(3, 3)):
    return closing(image, kernel_size) - image


# 形态学统一运算
def morphological_operations(image):
    # 应用各形态学运算得到输出结果
    erosion_image = erosion(image)
    dilation_image = dilation(image)
    opening_image = opening(image)
    closing_image = closing(image)
    tophat_image = tophat(image)
    blackhat_image = blackhat(image)

    # 返回结果
    return erosion_image, dilation_image, opening_image, closing_image, tophat_image, blackhat_image


def main():
    # 输入图像路径
    input_image_paths = ["src/origin_image.png", "src/gaussian_noise.png", "src/pepper_noise.png"]

    # 输入图片与滤波器标签
    imgs = ['origin', 'gaussian', 'pepper']
    filters = ["Erosion", "Dilation", "Opening", "Closing", "TopHat", "BlackHat"]

    for i, input_image_path in enumerate(input_image_paths):
        # 读取输入图像
        image = cv2.imread(input_image_path, 0)

        # 应用滤波器得到输出结果
        filtered_images = morphological_operations(image)

        # 保存结果, 例 3x3_Sobel_pepper
        for j in range(len(filters)):
            cv2.imwrite(os.path.join('task1_dst', imgs[i], f"3x3_{filters[j]}_{imgs[i]}.png"), filtered_images[j])


if __name__ == "__main__":
    main()
