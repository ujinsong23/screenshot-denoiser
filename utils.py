import numpy as np
from scipy.ndimage import sobel

def normalize(image):
    if image.dtype == np.uint8:
        return image / 255
    else:
        return (image-image.min()) / (image.max()-image.min())

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

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

def fill_foreground_patch(patch):
    patch_flatten = patch.flatten()
    var = np.var(patch_flatten)
    _, (counts, bin_edges) = compute_entropy(patch_flatten, return_counts=True)
    if var<0.01:
        max_count_index = np.argmax(counts)
        return bin_edges[max_count_index]
    else:
        return None

def cluster_colors(image, num_clusters=5):
    from sklearn.cluster import KMeans
    H, W, C = image.shape
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    quantized_image = centers[labels].reshape(H, W, C)
    if image.dtype == np.uint8:
        return quantized_image.astype(np.uint8), centers.astype(np.uint8)
    else:
        return quantized_image, centers

def compute_entropy(values, return_counts=False):
    if values.dtype != np.int8:
        values = (np.clip(values, 0, 1)*255).astype(np.uint8)
    counts, bin_edges = np.histogram(values, bins=256, range=(0, 256))
    
    total_counts = len(values)
    probabilities = counts / total_counts
    
    non_zero_probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(non_zero_probabilities * np.log(non_zero_probabilities))
    
    if return_counts:
        return entropy, (counts, bin_edges)
    else:
        return entropy

def remove_block_artifact(image):
    H, W = image.shape
    patch_size = 8
    stride = 1

    mask = image.copy()
    for i in range(0, H-patch_size, stride):
        for j in range(0, W-patch_size, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            if np.var(patch.flatten()) < 0.01:
                fill_value = fill_foreground_patch(patch)/255
                if fill_value is not None:
                    mask[i][j] = fill_value
    return mask