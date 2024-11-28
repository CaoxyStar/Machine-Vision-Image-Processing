"""
该程序为机器视觉课程滤波器作业 Task2, 主要功能为对图像进行频域处理。
作者: 曹翔宇
日期: 2024年4月10日
操作系统: Windows 11
环境: Python 3.9
依赖包: opencv-python, numpy
"""

import os

import cv2
import numpy as np


# 获取多个滤波器的频域幅值谱
def get_spectrum(img_size):
    # 3x3 x 方向 Sobel 滤波器
    sobel_x_filter = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    # 3x3 y 方向 Sobel 滤波器
    sobel_y_filter = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])

    # 3x3 高斯滤波器
    gaussian_filter = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]])

    # 3x3 拉普拉斯滤波器
    laplacian_filter = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])

    # 计算各滤波器根据图像尺寸补零后的幅值谱
    sobel_x_spec = filter_spectrum(sobel_x_filter, img_size)
    sobel_y_spec = filter_spectrum(sobel_y_filter, img_size)
    gaussian_spec = filter_spectrum(gaussian_filter, img_size)
    laplacian_spec = filter_spectrum(laplacian_filter, img_size)

    # 返回幅值谱
    return sobel_x_spec, sobel_y_spec, gaussian_spec, laplacian_spec


# 对3x3滤波器进行补零操作，然后求其幅值谱
def filter_spectrum(filter, img_size):
    # 补零后的尺寸
    height, width = img_size

    # 创建一个与图像尺寸相同的新数组，并用零填充
    padded_filter = np.zeros((height, width), dtype=np.float32)

    # 将高斯滤波器放在新数组的中心位置
    # center_row = height // 2
    # center_col = width // 2
    # padded_filter[center_row - 1:center_row + 2, center_col - 1:center_col + 2] = filter

    # 将高斯滤波器放在新数组的左上角
    h, w = filter.shape
    padded_filter[0:h, 0:w] = filter

    return magnitude_spectrum(padded_filter)


# 计算补零后滤波器的频谱
def magnitude_spectrum(image):
    # 对图像进行傅里叶变换
    f = np.fft.fft2(image)

    # 将频谱原点移至图像中心
    fshift = np.fft.fftshift(f)

    return fshift


# 将幅值谱调整到较好的可视化范围进行显示
def vis_spectrum(spec):
    return 20 * np.log(np.abs(spec) + 1e-10)


def main():
    # 输入图像读取路径
    input_image_paths = ["src/origin_image.png", "src/gaussian_noise.png", "src/pepper_noise.png"]

    # 输入图片的顺序标签
    img_labels = ['origin', 'gaussian', 'pepper']

    # 计算图像幅值谱
    for i in range(len(input_image_paths)):
        img = cv2.imread(input_image_paths[i], 0)
        spectrum = magnitude_spectrum(img)
        # 结果保存
        cv2.imwrite(os.path.join('task2_dst', 'raw_image_spectrum', f"{img_labels[i]}_spectrum.png"), vis_spectrum(spectrum))

    # 滤波器顺序标签及图像尺寸
    filter_labels = ["Sobel_x", "Sobel_y", "Gaussian", "Laplacian"]
    img_size = (374, 1238)

    # 计算滤波器按图像尺寸补零后的幅值谱
    filter_spectrums = get_spectrum(img_size)

    # 保存滤波器幅值谱
    for i in range(len(filter_labels)):
        cv2.imwrite(os.path.join('task2_dst', 'filter_spectrum', f"{filter_labels[i]}_spectrum.png"), vis_spectrum(filter_spectrums[i]))

    # 计算各滤波器与origin图像的幅值谱乘积后的结果, 然后进行傅里叶反变换得到滤波后的图像
    for i, filter_spectrum in enumerate(filter_spectrums):

        # 在频域进行origin图像与滤波器的幅值谱相乘
        origin_img = cv2.imread("src/origin_image.png", 0)
        origin_spec = magnitude_spectrum(origin_img)
        filtered_spectrum = origin_spec * filter_spectrum

        # 对乘积后的幅值谱进行傅里叶反变换
        filtered_img = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)))
        filtered_img = cv2.convertScaleAbs(filtered_img)

        # 保存结果
        cv2.imwrite(os.path.join('task2_dst', 'filtered_origin_spectrum_and_img', f"{filter_labels[i]}_filtered_origin_spectrum.png"), vis_spectrum(filtered_spectrum))
        cv2.imwrite(os.path.join('task2_dst', 'filtered_origin_spectrum_and_img', f"{filter_labels[i]}_filtered_origin_img.png"), filtered_img)

    # 计算各滤波器与gaussian图像的幅值谱乘积后的结果, 然后进行傅里叶反变换得到滤波后的图像
    for i, filter_spectrum in enumerate(filter_spectrums):

        # 在频域进行gaussian图像与滤波器的幅值谱相乘
        gaussian_img = cv2.imread("src/gaussian_noise.png", 0)
        gaussian_spec = magnitude_spectrum(gaussian_img)
        filtered_spectrum = gaussian_spec * filter_spectrum

        # 对乘积后的幅值谱进行傅里叶反变换
        filtered_img = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)))
        filtered_img = cv2.convertScaleAbs(filtered_img)

        # 保存结果
        cv2.imwrite(os.path.join('task2_dst', 'filtered_gaussian_spectrum_and_img', f"{filter_labels[i]}_filtered_gaussian_spectrum.png"), vis_spectrum(filtered_spectrum))
        cv2.imwrite(os.path.join('task2_dst', 'filtered_gaussian_spectrum_and_img', f"{filter_labels[i]}_filtered_gaussian_img.png"), filtered_img)


if __name__ == "__main__":
    main()
