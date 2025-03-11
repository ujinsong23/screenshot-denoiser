import numpy as np

def normalize(image):
    if image.dtype == np.uint8:
        return image / 255
    else:
        return (image-image.min()) / (image.max()-image.min())