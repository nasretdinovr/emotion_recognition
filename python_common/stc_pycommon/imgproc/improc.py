import math
import random
import numpy as np
from PIL import Image


def sigmoid(r):
    return 2 / (1 + np.exp(-r)) - 1


def compute_grad(nx, ny, A, C, a, b, alpha):
    ''' Generates brightness gradient
    :param nx: grid of normalized coordinates (x-coordinate)
    :param ny: grid of normalized coordinates (y-coordinate)
    :param A: variation amplitude
    :param C: value shift
    :param a: stretch factor
    :param b: center offset
    :param alpha: direction angle (in radians)
    :return: brightness gradient that should be added to the image
    '''
    r = (nx * np.cos(alpha) + ny * np.sin(alpha))
    return 255.0 * A * sigmoid(a * (r - b)) + C


def gen_params():
    A = 0.25 * 2 * (random.random() - 0.5)
    C = 0.25 * 2 * (random.random() - 0.5)
    a = 8 * random.random() + 2
    b = 0.6 * 2 * (random.random() - 0.5)
    alpha = math.pi * random.random()
    return A, C, a, b, alpha


def create_grid(shape, roi):
    [nx, ny] = np.meshgrid(range(shape[0]), range(shape[1]))
    center_x = roi[0] + 0.5 * (roi[2] - 1)
    center_y = roi[1] + 0.5 * (roi[3] - 1)
    nx = 2 * (nx - center_x) / (roi[2] - 1)
    ny = 2 * (ny - center_y) / (roi[3] - 1)
    return nx, ny


def variate_illumination(img, roi):
    A, C, a, b, alpha = gen_params()
    nx, ny = create_grid(img.shape[::-1], roi)
    grad = compute_grad(nx, ny, A, C, a, b, alpha)
    img += grad.astype(np.int64)
    img[img > 255] = 255
    img[img < 0] = 0
    return img
