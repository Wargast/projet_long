from matplotlib.pyplot import axis
import numpy as np
from tqdm import tqdm
import cv2
import time


def faster_transform(i: np.ndarray, block_size: int = 3) -> np.ndarray:
    if block_size % 2 == 0:
        raise ValueError("Block size must be an odd number")

    (h, l) = i.shape

    # only consider the pixels where the block doesn't go out of the image
    h_cropped = h - block_size + 1
    l_cropped = l - block_size + 1

    # get all blocks of the image of size (block_size x block_size)
    blocks = np.lib.stride_tricks.sliding_window_view(i, (block_size, block_size))

    blocks = blocks.reshape(h_cropped, l_cropped, block_size ** 2)
    return blocks

def faster_matching(
    i_left: np.ndarray,
    i_right: np.ndarray,
    block_size: int = 15,
    max_disparity: int = 64,
    cost_threshold: int = 125,
    metrique: str = 'ssd'
) -> np.ndarray:

    (h1, l1) = i_right.shape
    (h2, l2) = i_left.shape

    if h1 != h2:
        raise ValueError("Both images must have the same height")

    # only consider the pixels where the block doesn't go out of the image
    h1_cropped = h1 - block_size + 1
    l1_cropped = l1 - block_size + 1

    # calculate the bistrings for the first image :
    bit_strings_1 = faster_transform(i_right, block_size=block_size)

    l2_cropped = l2 - block_size + 1

    # calculate the bitstrings for the second image
    bit_strings_2 = faster_transform(i_left, block_size=block_size)

    # disparity_map = np.full((h1, l1), np.inf)

    error_map = np.full(
        (h1_cropped, l1_cropped, max_disparity + 1),
        np.inf,
    )

    for d in tqdm(range(max_disparity + 1)):
        right = bit_strings_1[:, 0 : min(l2_cropped - d, l1_cropped), :]
        left = bit_strings_2[:, d:, :]
        if metrique == 'ssd':
            costs = np.square(right-left).sum(axis=2)
        elif metrique == 'sad':
            costs = np.abs(right-left).sum(axis=2)
        error_map[:, 0 : min(l2_cropped - d, l1_cropped), d] = costs

    return error_map

def disparity_from_error_map(
    error_map: np.ndarray, block_size: int, cost_threshold: int = 125
) -> np.ndarray:
    (h_cropped, l_cropped, _) = error_map.shape
    half_block_size = block_size // 2

    errors = np.min(error_map, axis=2)
    disparity_map = np.argmin(error_map, axis=2).astype(np.float32)

    disparity_map[errors > cost_threshold] = np.inf

    disparity_map = np.pad(
        disparity_map,
        (half_block_size, half_block_size),
        mode='constant',
        constant_values=np.inf
    )

    return disparity_map


if __name__ == "__main__":
    i_left = cv2.imread("datas/middlebury/artroom1/im0.png", cv2.IMREAD_GRAYSCALE)
    i_right = cv2.imread("./datas/middlebury/artroom1/im1.png", cv2.IMREAD_GRAYSCALE)

    error_map = faster_matching(
        i_left, i_right, block_size=15, max_disparity=128
    )

    disparity_map = disparity_from_error_map(
        error_map, block_size=15, cost_threshold=1000000000
    )

    np.save("disp_ssd_artroom1.npy", disparity_map)