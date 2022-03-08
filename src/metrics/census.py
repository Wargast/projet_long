import numpy as np
from tqdm import tqdm
import census_c
import cv2
import time


def naive_census_transform(im: np.ndarray, block_size: int) -> np.ndarray:
    (h, l) = im.shape

    half_block_size = block_size // 2

    # only consider the pixels where the block doesn't go out of the image
    h_cropped = h - block_size + 1
    l_cropped = l - block_size + 1

    # calculate the bistrings for the image :
    bit_strings = np.zeros((h_cropped, l_cropped, block_size ** 2), dtype=np.bool_)

    for u in tqdm(range(h_cropped)):
        for v in range(l_cropped):
            block = im[
                u : u + block_size,
                v : v + block_size,
            ]

            center_value = block[half_block_size, half_block_size]

            # bit string : 1 if pixel value < center value of the block, else 0
            # for all pixels of the block
            bit_string = (block < center_value).reshape(1, block_size ** 2)

            bit_strings[u, v, :] = bit_string

    return bit_strings


def faster_census_transform(im: np.ndarray, block_size: int) -> np.ndarray:
    (h, l) = im.shape

    block_center_index = block_size ** 2 // 2

    # only consider the pixels where the block doesn't go out of the image
    h_cropped = h - block_size + 1
    l_cropped = l - block_size + 1

    # get all blocks of the image of size (block_size x block_size)
    blocks = np.lib.stride_tricks.sliding_window_view(im, (block_size, block_size))

    blocks = blocks.reshape(h_cropped, l_cropped, block_size ** 2)

    # get the value of the center of each block
    centers = blocks[:, :, block_center_index]

    centers = np.expand_dims(centers, axis=-1)

    # compare each block to its center to get the bit strings :
    # 1 if pixel value < center value, else 0
    bit_strings = blocks < centers

    return bit_strings


def naive_census_matching(
    im_left: np.ndarray, im_right: np.ndarray, block_size: int, max_disparity: int
) -> np.ndarray:

    (h1, l1) = im_left.shape
    (h2, l2) = im_right.shape

    if h1 != h2:
        raise ValueError("Both images must have the same height")

    if block_size % 2 == 0:
        raise ValueError("Block size must be an odd number")

    half_block_size = block_size // 2

    # only consider the pixels where the block doesn't go out of the image
    h1_cropped = h1 - block_size + 1
    l1_cropped = l1 - block_size + 1

    # calculate the bistrings for the first image :
    bit_strings_l = faster_census_transform(im_left, block_size)

    h2_cropped = h2 - block_size + 1
    l2_cropped = l2 - block_size + 1

    # calculate the bitstrings for the second image
    bit_strings_r = faster_census_transform(im_right, block_size)

    disparity_map = np.full((h1, l1), np.inf)

    # calculate the matching error between pixels of i1 and
    # pixels of i2 on the same line

    for u in tqdm(range(h1_cropped)):
        for v1 in range(l1_cropped):
            costs = np.full((max_disparity), np.inf)

            for d in range(max_disparity):
                v2 = v1 + d
                if v2 < l2_cropped:
                    bit_string_1 = bit_strings_l[u, v1]
                    bit_string_2 = bit_strings_r[u, v2]

                    # matching error : hamming distance between the two bistrings
                    cost = np.count_nonzero(bit_string_1 != bit_string_2)

                    costs[d] = cost

            min_cost = np.min(costs)
            if min_cost <= cost_threshold:
                disparity = np.argmin(costs)
                disparity_map[half_block_size + u, half_block_size + v1] = disparity

    return disparity_map


def faster_census_matching(
    im_left: np.ndarray, im_right: np.ndarray, block_size: int, max_disparity: int
) -> np.ndarray:

    (h1, l1) = im_right.shape
    (h2, l2) = im_left.shape

    if h1 != h2:
        raise ValueError("Both images must have the same height")

    if block_size % 2 == 0:
        raise ValueError("Block size must be an odd number")

    h1_cropped = h1 - block_size + 1
    l1_cropped = l1 - block_size + 1
    l2_cropped = l2 - block_size + 1

    bit_strings_l = faster_census_transform(im_left, block_size)
    bit_strings_r = faster_census_transform(im_right, block_size)

    error_map_l = np.full(
        (h1_cropped, l1_cropped, max_disparity + 1),
        np.iinfo(np.uint16).max,
        dtype=np.uint16,
    )

    error_map_r = np.full(
        (h1_cropped, l2_cropped, max_disparity + 1),
        np.iinfo(np.uint16).max,
        dtype=np.uint16,
    )

    for d in tqdm(range(max_disparity + 1)):
        right = bit_strings_r[:, 0 : min(l2_cropped - d, l1_cropped), :]
        left = bit_strings_l[:, d:, :]

        costs = np.sum(right != left, axis=2)

        error_map_r[:, 0 : min(l2_cropped - d, l1_cropped), d] = costs
        error_map_l[:, d:, d] = costs

    return error_map_l, error_map_r


def disparity_from_error_map(
    error_map_l: np.ndarray, error_map_r: np.ndarray, block_size: int, seuil_symmetrie: int
) -> np.ndarray:

    half_block_size = block_size // 2

    disparity_map_left = np.argmin(error_map_l, axis=2).astype(np.float32)
    disparity_map_right = np.argmin(error_map_r, axis=2).astype(np.float32)

    (h, l, _) = error_map_l.shape

    ind = np.meshgrid(range(h), range(l), indexing='ij')

    ind2 = (ind[0], (ind[1] - disparity_map_left).astype(np.uint16))

    d2 = disparity_map_right[ind2]

    disparity_map_left_filtered = disparity_map_left.copy()

    disparity_map_left_filtered[np.abs(d2 - disparity_map_left) > seuil_symmetrie] = np.inf

    disparity_map_left_filtered = np.pad(
        disparity_map_left_filtered,
        (half_block_size, half_block_size),
        mode="constant",
        constant_values=np.inf,
    )

    return disparity_map_left_filtered


if __name__ == "__main__":
    im_left = cv2.imread("./datas/middlebury/artroom1/im0.png", cv2.IMREAD_GRAYSCALE)
    im_right = cv2.imread("./datas/middlebury/artroom1/im1.png", cv2.IMREAD_GRAYSCALE)

    block_size = 9
    max_disparity = 64

    g = time.perf_counter()
    error_map_numpy_l, error_map_numpy_r = faster_census_matching(
        im_left, im_right, block_size, max_disparity
    )
    h = time.perf_counter()
    print(f"temps de calcul census matching numpy {h - g} s")

    a = time.perf_counter()
    error_map_cython_l, error_map_cython_r = census_c.census_matching(
        im_left, im_right, block_size, max_disparity
    )
    b = time.perf_counter()
    print(f"temps de calcul census matching cython {b - a} s")

    if np.array_equal(error_map_numpy_l, error_map_cython_l) and np.array_equal(error_map_numpy_r, error_map_cython_r):
        print("les cartes d'erreurs sont bien égales")
    else:
        raise ValueError("Les cartes d'erreurs n'ont pas la même valeur")


    c = time.perf_counter()
    disp_map_filtered = disparity_from_error_map(error_map_cython_l, error_map_cython_r, block_size, seuil_symmetrie=5)
    d = time.perf_counter()
    print(f"temps de calcul disparités numpy {d - c} s")

    e = time.perf_counter()
    disp_map_cython_filtered = census_c.disparity_from_error_map(error_map_cython_l, error_map_cython_r, block_size, seuil_symmetrie=5)
    f = time.perf_counter()
    print(f"temps de calcul disparités cython {f - e} s")

    if np.array_equal(disp_map_filtered, disp_map_cython_filtered):
        print("les cartes de disparités sont bien égales")
    else:
        raise ValueError("Les cartes de disparité n'ont pas la même valeur")



    # print(np.array_equal(error_map_cython_l, error_map_numpy_l))
    # print(np.array_equal(error_map_cython_r, error_map_numpy_r))
