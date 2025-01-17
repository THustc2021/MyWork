import os

import torch
import torch.nn.functional as F

# IMG_SIZE = (488, 712)
IMG_SIZE = (448, 448)
GRID_SIZE = (45, 31)

def get_version():
    """
    Returns the version of the various packages used for evaluation.
    """
    import pytesseract

    return {
        "tesseract": str(pytesseract.get_tesseract_version()),
        "pyesseract": os.popen("pip list | grep pytesseract").read().split()[-1],
        "Levenshtein": os.popen("pip list | grep Levenshtein").read().split()[-1],
        "jiwer": os.popen("pip list | grep jiwer").read().split()[-1],
        "matlabengineforpython": os.popen("pip list | grep matlab").read().split()[-1],
    }


def bilinear_unwarping(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Args:
        warped_img  :       torch.Tensor of shape BxCxHxW (dtype float)
        point_positions:    torch.Tensor of shape Bx2xGhxGw (dtype float)
        img_size:           tuple of int [w, h]
    """
    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True)

    return unwarped_img, upsampled_grid


def bilinear_unwarping_from_numpy(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Accept numpy arrays as input.
    """
    warped_img = torch.unsqueeze(torch.from_numpy(warped_img.transpose(2, 0, 1)).float(), dim=0)
    point_positions = torch.unsqueeze(torch.from_numpy(point_positions.transpose(2, 0, 1)).float(), dim=0)

    unwarped_img = bilinear_unwarping(warped_img, point_positions, img_size)

    unwarped_img = unwarped_img[0].numpy().transpose(1, 2, 0)
    return unwarped_img