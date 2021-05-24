#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2019-12-26 14:56:08
@LastEditTime: 2020-06-08 11:13:27
'''
from fhog import fHOG
import numpy as np
import cv2
img = cv2.imread("./fhog/lena.png", cv2.IMREAD_COLOR)
img = np.float32(img)/255.0
print(img.shape)
hog = fHOG(img)
for c in range(hog.shape[-1]):
    tmp = hog[:, :, c].copy()
    tmp = cv2.normalize(tmp, tmp, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite("hog_%d.jpg"%c, tmp)
print(hog.shape)