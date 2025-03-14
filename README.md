# screenshot-denoiser
*2024-25Winter - Stanford EE368*

This repository contains the implementation of a denoising framework designed to reduce JPEG artifacts in digital screenshots, particularly focusing on enhancing text clarity and preserving important details.

## Files

- `dataset_generate.py`: Generates a dataset based on the screenshot of your choice, ensuring they are uniformly sized at 600x600 pixels.

- `denoiser.py`: Includes our two types of filters and allows the user to input the image path, set hyperparameters, and save the denoised image. Run this script as follows:
```bash
python denoiser.py --image_id 7 --sigma_r 0.3 --sigma_s 5 --sigma_o 0.1 --eps 0.0002
```

- `utils.py`: Contains utility functions for computing MSE loss, image normalization to (0,1), gradient computation using Sobel filters, etc.

- `notebook.ipynb`: A Jupyter notebook that utilizes the functions defined above, mainly for creating plots for the final report.