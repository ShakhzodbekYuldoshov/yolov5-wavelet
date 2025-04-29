# wavelet transform layer noise reduction

import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt


def wavelet_denoising(image):
    # 将图像进行小波变换
    coeffs = pywt.dwt2(image, 'bior1.3')
    cA, (cH, cV, cD) = coeffs
    # 将高频系数置零
    cH = np.zeros(cH.shape)
    cV = np.zeros(cV.shape)
    cD = np.zeros(cD.shape)
    # 将图像进行小波逆变换
    coeffs = cA, (cH, cV, cD)
    image = pywt.idwt2(coeffs, 'bior1.3')
    return image

image = cv2.imread('image.jpg', 0)
image = wavelet_denoising(image)
plt.imshow(image, cmap='gray')
plt.show()