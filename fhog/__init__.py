from .fhog import gradientMag, gradientHist
import numpy as np
def fHOG(im_patch):
    M = np.zeros(im_patch.shape[:2], dtype='float32')
    O = np.zeros(im_patch.shape[:2], dtype='float32')
    H = np.zeros([im_patch.shape[0]//4,im_patch.shape[1]//4, 32], dtype='float32') # python3
    gradientMag(im_patch.astype(np.float32),M,O)
    gradientHist(M,O,H)
    return H[:, :, :31]

__all__ = ['fHOG']
