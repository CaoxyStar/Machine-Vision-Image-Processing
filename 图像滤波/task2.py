"""
该程序为机器视觉课程滤波器作业 Task2, 主要功能为比较双边、引导、均值、高斯滤波器处理图片的效果，其中双边、引导滤波器为手动实现。
作者: 曹翔宇
日期: 2024年4月7日
操作系统: Windows 11
环境: Python 3.9
依赖包: opencv-python, opencv-contrib-python, numpy
"""

import os
import cv2

import numpy as np


# 高斯权重计算（简化掉常数项系数）
def gaussian(x_square, sigma):
    return np.exp(-0.5*x_square/sigma**2)


# 单通道快速双边滤波
def single_bilateral_filter(image, sigma_space, sigma_intensity):
    # kernel_size = sigma_space * 2 + 1
    kernel_size = int(2 * sigma_space + 1)
    half_kernel_size = int(kernel_size / 2)
    # 初始化返回图片
    result = np.zeros(image.shape)
    # 权重和
    w = 0

    # 加速滤波过程
    for x in range(-half_kernel_size, half_kernel_size+1):
        for y in range(-half_kernel_size, half_kernel_size+1):
            # 计算空间权重系数
            Gspace = gaussian(x ** 2 + y ** 2, sigma_space)
            shifted_image = np.roll(image, [x, y], [1, 0])
            # 计算像素权重系数
            intensity_difference_image = image - shifted_image
            Gintenisity = gaussian(
                intensity_difference_image ** 2, sigma_intensity)
            # 加权和
            result += Gspace*Gintenisity*shifted_image
            w += Gspace*Gintenisity
    # 除以总的权重和
    return result / w


# 多通道双边滤波
def multi_bilateral_filter(input_image, sigma_space=3, sigma_intensity=0.5):
    # 图像归一化处理，方便计算
    input_image = input_image / 255.0

    # 各通道单独进行滤波
    bf = single_bilateral_filter(input_image[:, :, 0], sigma_space, sigma_intensity)
    gf = single_bilateral_filter(input_image[:, :, 1], sigma_space, sigma_intensity)
    rf = single_bilateral_filter(input_image[:, :, 2], sigma_space, sigma_intensity)

    # 将经过滤波后的各通道合并，并放缩回原像素分布
    output_image = np.stack([bf, gf, rf], axis=2)
    return output_image * 255.0


# 单通道引导滤波
def guided_filter(I, p, radius=7, eps=0.1):
    # 用均值滤波器对输入图像 I 和导向图像 p 进行滤波
    mean_I = cv2.boxFilter(I, -1, (radius, radius))
    mean_p = cv2.boxFilter(p, -1, (radius, radius))

    # 用协方差滤波器对 I*p 进行滤波
    mean_Ip = cv2.boxFilter(I * p, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    # 计算 p 的方差和 I*p 的协方差
    mean_II = cv2.boxFilter(I * I, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    # 计算 a 和 b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 用均值滤波器对 a 和 b 进行滤波
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    # 计算输出 q
    q = mean_a * I + mean_b
    return q


# 多通道引导滤波
def guided_filter_color(image, radius=7, eps=0.1):
    # 将图像转换为浮点数类型，并归一化到 [0, 1] 范围
    image = np.float32(image) / 255.0

    # 将每个通道像素作为每个通道的引导图片
    guide = cv2.split(image)

    # 应用引导滤波器到每个通道
    filtered_channels = []
    for i in range(image.shape[2]):
        filtered_channels.append(guided_filter(image[:, :, i], guide[i], radius, eps))

    # 合并每个通道的结果
    filtered_image = np.stack(filtered_channels, axis=2)

    return (filtered_image * 255).astype(np.uint8)


# 定义不同的滤波器
def apply_filters(image):
    # 7x7 双边滤波，空间标准差为3，像素标准差为0.5
    bilateral_res = multi_bilateral_filter(image, sigma_space=3, sigma_intensity=0.5)

    # 7x7 引导滤波，eps 0.1
    guide_res = guided_filter_color(image, radius=7, eps=0.1)

    # 7x7 均值滤波
    mean_blur = cv2.blur(image, (7, 7), borderType=cv2.BORDER_REPLICATE)

    # 7x7 高斯滤波，sigmaX 1.4，sigmaY 1.4
    gaussian_blur = cv2.GaussianBlur(image, (7, 7), sigmaX=1.4, sigmaY=1.4, borderType=cv2.BORDER_REPLICATE)

    return bilateral_res, guide_res, mean_blur, gaussian_blur


def main():
    # 输入图像读取路径
    input_image_paths = ["src/origin_image.png", "src/gaussian_noise.png", "src/pepper_noise.png"]

    # 输入图片及滤波器标签
    imgs = ['origin', 'gaussian', 'pepper']
    filters = ["Bilateral", "Guide", "Mean", "Gaussian"]

    for i, input_image_path in enumerate(input_image_paths):
        # 读取输入图像
        image = cv2.imread(input_image_path)

        # 应用滤波器
        filtered_images = apply_filters(image)

        # 保存结果，例 7x7_Bilateral_pepper
        for j in range(len(filters)):
            cv2.imwrite(os.path.join('task2_dst', imgs[i], f"7x7_{filters[j]}_{imgs[i]}.png"), filtered_images[j])


if __name__ == "__main__":
    main()
