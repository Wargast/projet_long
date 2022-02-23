import numpy as np
from tqdm import tqdm
import census_c
import cv2
import time
# import pprofile


def naive_census_transform(i: np.ndarray, block_size: int = 3) -> np.ndarray:
    if block_size % 2 == 0:
        raise ValueError("Block size must be an odd number")

    (h, l) = i.shape

    half_block_size = block_size // 2

    # only consider the pixels where the block doesn't go out of the image
    h_cropped = h - block_size + 1
    l_cropped = l - block_size + 1

    # calculate the bistrings for the image :
    bit_strings = np.zeros((h_cropped, l_cropped, block_size ** 2), dtype=np.bool_)

    for u in tqdm(range(h_cropped)):
        for v in range(l_cropped):
            block = i[
                u : u + block_size,
                v : v + block_size,
            ]

            center_value = block[half_block_size, half_block_size]

            # bit string : 1 if pixel value < center value of the block, else 0
            # for all pixels of the block
            bit_string = (block < center_value).reshape(1, block_size ** 2)

            bit_strings[u, v, :] = bit_string

    return bit_strings


def faster_census_transform(i: np.ndarray, block_size: int = 3) -> np.ndarray:
    if block_size % 2 == 0:
        raise ValueError("Block size must be an odd number")

    (h, l) = i.shape

    block_center_index = block_size ** 2 // 2

    # only consider the pixels where the block doesn't go out of the image
    h_cropped = h - block_size + 1
    l_cropped = l - block_size + 1

    # get all blocks of the image of size (block_size x block_size)
    blocks = np.lib.stride_tricks.sliding_window_view(i, (block_size, block_size))

    blocks = blocks.reshape(h_cropped, l_cropped, block_size ** 2)

    # get the value of the center of each block
    centers = blocks[:, :, block_center_index]

    centers = np.expand_dims(centers, axis=-1)

    # compare each block to its center to get the bit strings :
    # 1 if pixel value < center value, else 0
    bit_strings = blocks < centers

    return bit_strings


def naive_census_matching(
    i_left: np.ndarray,
    i_right: np.ndarray,
    block_size: int = 15,
    max_disparity: int = 64,
    cost_threshold: int = 150,
) -> np.ndarray:

    (h1, l1) = i_left.shape
    (h2, l2) = i_right.shape

    if h1 != h2:
        raise ValueError("Both images must have the same height")

    half_block_size = block_size // 2

    # only consider the pixels where the block doesn't go out of the image
    h1_cropped = h1 - block_size + 1
    l1_cropped = l1 - block_size + 1

    # calculate the bistrings for the first image :
    bit_strings_1 = faster_census_transform(i_left, block_size=block_size)

    h2_cropped = h2 - block_size + 1
    l2_cropped = l2 - block_size + 1

    # calculate the bitstrings for the second image
    bit_strings_2 = faster_census_transform(i_right, block_size=block_size)

    disparity_map = np.full((h1, l1), np.inf)

    # calculate the matching error between pixels of i1 and
    # pixels of i2 on the same line

    for u in tqdm(range(h1_cropped)):
        for v1 in range(l1_cropped):
            costs = np.full((max_disparity), np.inf)

            for d in range(max_disparity):
                v2 = v1 + d
                if v2 < l2_cropped:
                    bit_string_1 = bit_strings_1[u, v1]
                    bit_string_2 = bit_strings_2[u, v2]

                    # matching error : hamming distance between the two bistrings
                    cost = np.count_nonzero(bit_string_1 != bit_string_2)

                    costs[d] = cost

            min_cost = np.min(costs)
            if min_cost <= cost_threshold:
                disparity = np.argmin(costs)
                disparity_map[half_block_size + u, half_block_size + v1] = disparity

    return disparity_map


def faster_census_matching(
    bit_strings_l: np.ndarray,
    bit_strings_r: np.ndarray,
    h1_cropped: int,
    l1_cropped: int,
    l2_cropped: int,
    block_size: int = 15,
    max_disparity: int = 64
) -> np.ndarray:

    (h1, l1) = i_right.shape
    (h2, l2) = i_left.shape

    if h1 != h2:
        raise ValueError("Both images must have the same height")

    # only consider the pixels where the block doesn't go out of the image
    h1_cropped = h1 - block_size + 1
    l1_cropped = l1 - block_size + 1

    # calculate the bistrings for the first image :
    bit_strings_1 = faster_census_transform(i_right, block_size=block_size)

    l2_cropped = l2 - block_size + 1

    # calculate the bitstrings for the second image
    bit_strings_2 = faster_census_transform(i_left, block_size=block_size)

    # disparity_map = np.full((h1, l1), np.inf)

    error_map = np.full(
        (h1_cropped, l1_cropped, max_disparity + 1),
        np.iinfo(np.uint16).max,
        dtype=np.uint16,
    )

    for d in tqdm(range(max_disparity + 1)):
        right = bit_strings_1[:, 0 : min(l2_cropped - d, l1_cropped), :]
        left = bit_strings_2[:, d:, :]

        error_map[:, 0 : min(l2_cropped - d, l1_cropped), d] = np.sum(right != left, axis=2)

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
    i_left = cv2.imread("./datas/middlebury/artroom1/im0.png", cv2.IMREAD_GRAYSCALE)

    i_right = cv2.imread("./datas/middlebury/artroom1/im1.png", cv2.IMREAD_GRAYSCALE)

    (h1, l1) = i_right.shape
    (h2, l2) = i_left.shape

    block_size = 9

    max_disparity = 64

    h1_cropped = h1 - block_size + 1
    l1_cropped = l1 - block_size + 1
    l2_cropped = l2 - block_size + 1

    bs_r = faster_census_transform(i_right, block_size=block_size)
    bs_l = faster_census_transform(i_left, block_size=block_size)
    
    a = time.perf_counter()
    error_map = census_c.census_matching(bs_l, bs_r, h1_cropped, l1_cropped, l2_cropped, block_size=block_size, max_disparity=max_disparity)
    b = time.perf_counter()
    print(f"temps cython : {b - a}")
    
    error_map2 = faster_census_matching(bs_l, bs_r, h1_cropped, l1_cropped, l2_cropped, block_size=block_size, max_disparity=max_disparity)
    c = time.perf_counter()
    print(f"temps numpy : {c - b}")

    print(np.array_equal(error_map, error_map2))

    
    

    # prof = pprofile.Profile()

    # with prof:
    #     error_map = faster_census_matching(
    #         i_left, i_right, block_size=15, max_disparity=150
    #     )

    #     disparity_map = disparity_from_error_map(
    #         error_map, block_size=15, cost_threshold=100
    #     )

    # prof.dump_stats("profiling.txt")

    # # np.save("disp_census_artroom1.npy", disparity_map)
    # np.save("error_map_census_artroom1.npy", error_map)
