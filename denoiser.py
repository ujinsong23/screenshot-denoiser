import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from scipy.ndimage import uniform_filter
DATA_PATH = "../dataset"
from utils import normalize, rgb2gray, compute_gradients, mse, psnr, sigmoid

def bilateral_get_weight(i_range, j_range, i, j, p, gradient_magnitudes, sigma_s, sigma_r, sigma_o):
    patch = p[np.ix_(i_range, j_range)]
    center_gradient = gradient_magnitudes[i, j]
    neighbor_gradient = gradient_magnitudes[np.ix_(i_range, j_range)]
    w = np.exp(
        - ((patch - p[i, j])**2 / sigma_r**2)
        - ((i_range[:, None] - i)**2 / sigma_s**2)
        - ((j_range[None, :] - j)**2 / sigma_s**2)
    )

    orientation_distance = (np.linalg.norm(neighbor_gradient - center_gradient, axis=-1)**2)*np.linalg.norm(center_gradient)
    w_orientation = np.exp(-orientation_distance/sigma_o**2)
    return w, w_orientation, w*w_orientation

def bilateral_filter(p, r, gradient_magnitudes, sigma_s, sigma_r, sigma_o):
    H, W = p.shape
    q = np.zeros((H, W))
    for i in tqdm(range(H)):
        for j in range(W):
            i_range = np.arange(max(i - r, 0), min(i + r, H))
            j_range = np.arange(max(j - r, 0), min(j + r, W))
            patch = p[np.ix_(i_range, j_range)]            
            w = bilateral_get_weight(i_range, j_range, i, j, p, gradient_magnitudes, sigma_s, sigma_r, sigma_o)[-1]

            q[i, j] = np.sum(w * patch) / np.sum(w)
    return q

def adaptive_eps(variance, max_eps):
    sigmoid_var = sigmoid((variance-0.010)/0.1)
    adaptive_eps_map = max_eps * (1 - sigmoid_var)
    return adaptive_eps_map

def guided_filter(I, p, r, eps):
    mean_I = uniform_filter(I, size=r)
    mean_p = uniform_filter(p, size=r)
    
    corr_I = uniform_filter(I * I, size=r)
    corr_Ip = uniform_filter(I * p, size=r)
    
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    adaptive_eps_map = adaptive_eps(var_I, eps)
    
    a = cov_Ip / (var_I + adaptive_eps_map)
    b = mean_p - a * mean_I
    
    mean_a = uniform_filter(a, size=r)
    mean_b = uniform_filter(b, size=r)
    
    q = mean_a * I + mean_b
    return q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id", type=int, default=8)
    parser.add_argument("--sigma_r", type=float, default=0.3)
    parser.add_argument("--sigma_s", type=float, default=5)
    parser.add_argument("--sigma_o", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=2*1e-4)
    args = parser.parse_args()

    sigma_r = args.sigma_r
    sigma_s = args.sigma_s
    sigma_o = args.sigma_o
    eps = args.eps
    image_id = args.image_id
    print(f'Processing image {image_id}')
    print(f'Hyperparameters: sigma_r={sigma_r}, sigma_s={sigma_s}, sigma_o={sigma_o}, eps={eps}')

    clean = np.array(Image.open(f"{DATA_PATH}/original/{image_id:03d}.png"))[:, :, :3]
    noisy = np.array(Image.open(f"{DATA_PATH}/noised/{image_id:03d}.jpg"))
    clean = rgb2gray(normalize(clean))
    p = rgb2gray(normalize(noisy))
    print(f'Before denoising MSE: {mse(clean, p)}, PSNR: {psnr(clean, p)}')

    magnitudes = compute_gradients(p, num_directions=2)
    I = bilateral_filter(p, gradient_magnitudes=magnitudes, r=4,
                        sigma_s=sigma_s, sigma_r=sigma_r, sigma_o=sigma_o)
    q = guided_filter(I=I, p=p, r=3, eps=eps)

    print(f"After denoising MSE: {mse(clean, q)}, PSNR: {psnr(clean, q)}")

    denoised = (np.clip(q, 0, 1)*255).astype(np.uint8)
    denoised_image = Image.fromarray(denoised).save(f"{DATA_PATH}/denoised/{image_id:03d}.png")