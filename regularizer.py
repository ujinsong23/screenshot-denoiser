import torch
import torch.nn.functional as F

def tv_norm(x):
    if x.ndim==2:
        x = x.unsqueeze(0).unsqueeze(0)
    h, w = x.shape[2], x.shape[3]
    diff1 = x[:, 1:, :] - x[:, :-1, :]
    diff2 = x[:, :, 1:] - x[:, :, :-1]
    tv = torch.sum(torch.sqrt(diff1 ** 2 + 1e-8)) + torch.sum(torch.sqrt(diff2 ** 2 + 1e-8))
    tv = tv / (h * w)
    return tv #0.1158

def laplacian3(x):
    if x.ndim==3:
        assert x.shape[-1]==3, "input shape must be (H, W, 3) but got {}".format(x.shape)
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    elif x.ndim==4:
        assert x.shape[1]==3, "input shape must be (B, 3, H, W) but got {}".format(x.shape)
    else:
        raise ValueError("input shape must be (H, W, 3) or (B, 3, H, W) but got {}".format(x.shape))

    kernel = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float32).expand(1, 3, -1, -1)/3
    padding = kernel.shape[-1] // 2
    response_laplacian = (
        F.conv2d(x, kernel, padding=padding)
        .squeeze()
        # .squeeze().detach().numpy()
        )
    return response_laplacian

def laplacian5(x):
    laplacian_kernel = torch.tensor([[[[0,  0, -1,  0,  0],
                                       [0, -1, -2, -1,  0],
                                       [-1, -2, 16, -2, -1],
                                       [0, -1, -2, -1,  0],
                                       [0,  0, -1,  0,  0]]]], dtype=torch.float32)

    lap = 0
    for c in range(3):
        edges = F.conv2d(x[:, c, :, :].unsqueeze(1), laplacian_kernel, padding=2)
        lap += torch.mean(torch.abs(edges))
    return -lap

def differenceOfSigmoid(edge, c1=2, c2=5):
    r1 = F.sigmoid(edge*c1)
    r2 = F.sigmoid(edge*c2)
    return r1, r2, (r1-r2).abs().mean()