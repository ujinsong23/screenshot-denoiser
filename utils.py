import numpy as np
from scipy.ndimage import sobel

def normalize(image):
    if image.dtype == np.uint8:
        return image / 255
    else:
        return (image-image.min()) / (image.max()-image.min())

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_gradients(image, num_directions=2):
    assert num_directions in [2, 4]
    grad_x = sobel(image, axis=0)  # Gradient in x-direction
    grad_y = sobel(image, axis=1)  # Gradient in y-direction

    if num_directions==2:
        return np.stack((np.abs(grad_x), np.abs(grad_y)), axis=-1)
    else:
        grad_45 = sobel(image, axis=-1)  # Gradient in 45-degree direction
        grad_135 = sobel(image, axis=-2)  # Gradient in 135-degree direction
        return np.stack((np.abs(grad_x), np.abs(grad_y), np.abs(grad_45), np.abs(grad_135)), axis=-1)

def mse(image1, image2):
    return np.mean((image1 - image2)**2)

def psnr(clean, denoised):
    mse_value = mse(clean, denoised)
    if mse_value ==0:
        return float("inf")
    max_pixel = 1.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value