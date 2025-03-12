import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from scipy.ndimage import uniform_filter
DATA_PATH = "../dataset"
from utils import normalize, rgb2gray, compute_gradients, mse, psnr

def bilateral_get_weight(i_range, j_range, i, j, p, gradient_magnitudes, sigma_s, sigma_r, sigma_o):
    patch = p[np.ix_(i_range, j_range)]
    center_gradient = gradient_magnitudes[i, j]
    neighbor_gradient = gradient_magnitudes[np.ix_(i_range, j_range)]
    w = np.exp(
        - ((patch - p[i, j])**2 / sigma_r**2)
        - ((i_range[:, None] - i)**2 / sigma_s**2)
        - ((j_range[None, :] - j)**2 / sigma_s**2)
    )

    # w_orientation = np.exp(
    #     - ((neighbor_gradient[:,:,0] - center_gradient[0])**2 / sigma_o**2)
    #     - ((neighbor_gradient[:,:,1] - center_gradient[1])**2 / sigma_o**2)
    # )

    orientation_distance = (np.linalg.norm(neighbor_gradient - center_gradient, axis=-1)**2)*np.linalg.norm(center_gradient)
    w_orientation = np.exp(-orientation_distance/sigma_o**2)

    # def softmax(x):
    #     exp_x = np.exp(x - np.max(x))
    #     return exp_x / exp_x.sum()
    # neighbor_gradient /= (np.linalg.norm(neighbor_gradient, axis=-1, keepdims=True)+1e-6)
    # w_orientation = neighbor_gradient.dot(center_gradient.reshape(-1, 1)).squeeze()
    # w_orientation = softmax(w_orientation)

    # dot_product = np.sum(neighbor_gradient * center_gradient, axis=-1)
    # norms = np.linalg.norm(neighbor_gradient, axis=-1) * np.linalg.norm(center_gradient)
    # orientation_similarity /= norms

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
    # normalized_variance = (variance - np.min(variance)) / (np.max(variance) - np.min(variance))
    # adaptive_eps_map = max_eps-max_eps * normalized_variance


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # norm_var = (variance - np.mean(variance)) / (np.std(variance) + 1e-12)
    # sigmoid_var = sigmoid(norm_var/t)
    # adaptive_eps_map = max_eps * (1 - sigmoid_var) 
    sigmoid_var = sigmoid((variance-0.010)/0.1)
    adaptive_eps_map = max_eps * (1 - sigmoid_var)

    # def softmax(x):
    #     exp_x = np.exp(x - np.max(x))
    #     return exp_x / exp_x.sum()
    # norm_var = (variance - np.mean(variance)) / (np.std(variance) + 1e-12)
    # softmax_var = softmax(norm_var.flatten()/t).reshape(variance.shape)
    # adaptive_eps_map = max_eps * (1 - softmax_var) 
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
    # a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = uniform_filter(a, size=r)
    mean_b = uniform_filter(b, size=r)
    
    q = mean_a * I + mean_b
    return q


if __name__ == "__main__":
    # read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--sigma_r", type=float, default=0.3)
    parser.add_argument("--sigma_s", type=float, default=5)
    parser.add_argument("--sigma_o", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=2*1e-4)
    args = parser.parse_args()

    sigma_r = 0.3
    sigma_s = 5
    sigma_o = 0.1
    eps = 2*1e-4

    image_id = 6
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