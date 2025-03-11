import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import normalize
from regularizer import *
import torch
import torch.nn.functional as F
DATA_PATH = "../dataset"


if __name__ == "__main__":

    image_id = 6
    original = np.array(Image.open(f"{DATA_PATH}/original/{image_id:03d}.png"))[:, :, :3]
    noised = np.array(Image.open(f"{DATA_PATH}/noised/{image_id:03d}.jpg"))

    original = normalize(original)
    noised = normalize(noised)
    
    iterations = 200
    learning_rate = 1
    alpha = 5
    
    original = torch.tensor(original, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # (1, 3, 600, 600)
    noised = torch.tensor(noised, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # (1, 3, 600, 600)
    denoised = noised.clone().detach().requires_grad_(True)
    
    print(f'initial difference: {F.mse_loss(denoised, original).item():.06f}')

    pbar = tqdm(range(iterations))
    for i in pbar:
        mse = F.mse_loss(denoised, noised)
        
        edge = laplacian3(denoised)
        r1, r2, diff_sigmoid = differenceOfSigmoid(edge, c1=2, c2=10)
        regularizer = 0 * diff_sigmoid + tv_norm(r2)
        loss = mse + alpha * regularizer
        
        loss.backward()


        with torch.no_grad():
            denoised -= learning_rate * denoised.grad
            criteria = F.mse_loss(denoised, original)
            pbar.set_description(f"Loss: {mse.item():.6f}+{regularizer:.6f}, criteria: {criteria.item():.6f}, grad: {denoised.grad.abs().mean().item():.6f}")

        denoised.grad.zero_()

    denoised_image = denoised.squeeze(0).permute(1,2,0).detach().numpy()
    denoised_image = np.clip(denoised_image, 0, 1)
    denoised_image = (denoised_image * 255).astype(np.uint8)
    denoised_image = Image.fromarray(denoised_image).save(f"{DATA_PATH}/denoised/{image_id:03d}.png")