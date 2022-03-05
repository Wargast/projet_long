from re import I
import numpy as np
from tqdm import tqdm
import census_c
import cv2
import time
import matplotlib.pyplot as plt


class Local_matching:
    def __init__(
        self,
        max_disparity: int = 64,
        block_size: int = 9,
        method: str = "census",
        optimisation = "cython",
        seuil_symmetrie: int = 0,
        cost_threshold: int = 1000
    ):
        """__init__ instance de la classe local matching

        Args:
            max_disparity (int, optional): [valeur maximal de matching]. Defaults to 64.
            block_size (int, optional): [taille du bloc de correlation]. Defaults to 9.
            methode (str, optional): [methode utilisée (census/ssd/sad)]. Defaults to "census".
            optimisation (str,optional) : [optimisation utiliser (cython/python/python_naif). Defaults to "cython]
        """
        self.method = method
        self.max_disparity = max_disparity
        self.block_size = block_size
        self.seuil_symmetrie = seuil_symmetrie
        self.optimisation = optimisation
        self.cost_threshold = cost_threshold


    def compute(self, i_left: np.ndarray, i_right: np.ndarray):
        if self.optimisation == "cython":
            error_map_l, error_map_r = census_c.census_matching(i_left, i_right, self.block_size, self.max_disparity)
            return self._disparity_from_error_map2(error_map_l, error_map_r)
        elif self.optimisation == "python":
            error_map = self._faster_matching(i_left, i_right)
        elif self.optimisation == "python_naif":
            error_map = self._naive_matching(i_left,i_right)
        return self._disparity_from_error_map(error_map)


    def _disparity_from_error_map2(self,error_map_l: np.ndarray,error_map_r: np.ndarray,) -> np.ndarray:
        return census_c.disparity_from_error_map(error_map_l, error_map_r, self.block_size, self.seuil_symmetrie
)

    def _disparity_from_error_map(self,error_map: np.ndarray) -> np.ndarray:
        (h_cropped, l_cropped, _) = error_map.shape
        half_block_size = self.block_size // 2

        errors = np.min(error_map, axis=2)
        disparity_map = np.argmin(error_map, axis=2).astype(np.float32)

        disparity_map[errors > self.cost_threshold] = np.inf

        disparity_map = np.pad(
            disparity_map,
            (half_block_size, half_block_size),
            mode='constant',
            constant_values=np.inf
        )

        return disparity_map
   
    def _naive_transform(self,i: np.ndarray) -> np.ndarray:
        if self.block_size % 2 == 0:
            raise ValueError("Block size must be an odd number")
        (h, l) = i.shape

        half_block_size = self.block_size // 2

        # only consider the pixels where the block doesn't go out of the image
        h_cropped = h - self.block_size + 1
        l_cropped = l - self.block_size + 1

        # calculate the bistrings for the image :
        bit_strings = np.zeros(
            (h_cropped, l_cropped, self.block_size ** 2),
            dtype=np.bool_
        )

        for u in tqdm(range(h_cropped)):
            for v in range(l_cropped):
                block = i[
                    u : u + self.block_size,
                    v : v + self.block_size,
                ]

                center_value = block[half_block_size, half_block_size]

                # bit string : 1 if pixel value < center value of the block, else 0
                # for all pixels of the block
                bit_string = (block < center_value).reshape(1, self.block_size ** 2)

                bit_strings[u, v, :] = bit_string
        return bit_strings

    def _faster_transform(self,i: np.ndarray) -> np.ndarray:
        if self.block_size % 2 == 0:
            raise ValueError("Block size must be an odd number")

        (h, l) = i.shape

        block_center_index = self.block_size ** 2 // 2

        # only consider the pixels where the block doesn't go out of the image
        h_cropped = h - self.block_size + 1
        l_cropped = l - self.block_size + 1

        # get all blocks of the image of size (block_size x block_size)
        blocks = np.lib.stride_tricks.sliding_window_view(
            i, 
            (self.block_size, self.block_size)
        )

        blocks = blocks.reshape(h_cropped, l_cropped, self.block_size ** 2)
        bit_strings = blocks
        if self.methode == "census":
            # get the value of the center of each block
            centers = blocks[:, :, block_center_index]

            centers = np.expand_dims(centers, axis=-1)

            # compare each block to its center to get the bit strings :
            # 1 if pixel value < center value, else 0
            bit_strings = blocks < centers
        return bit_strings

    def _naive_matching(self,i_left: np.ndarray,i_right: np.ndarray,) -> np.ndarray:

        (h1, l1) = i_left.shape
        (h2, l2) = i_right.shape

        if h1 != h2:
            raise ValueError("Both images must have the same height")

        half_block_size = self.block_size // 2

        # only consider the pixels where the block doesn't go out of the image
        h1_cropped = h1 - self.block_size + 1
        l1_cropped = l1 - self.block_size + 1

        # calculate the bistrings for the first image :
        bit_strings_1 = self._faster_transform(i_left)
        
        h2_cropped = h2 - self.block_size + 1
        l2_cropped = l2 - self.block_size + 1

        # calculate the bitstrings for the second image
        bit_strings_2 = self._faster_transform(i_right)

        disparity_map = np.full((h1, l1), np.inf)

        # calculate the matching error between pixels of i1 and
        # pixels of i2 on the same line
        if self.methode == "census":
            for u in tqdm(range(h1_cropped)):
                for v1 in range(l1_cropped):
                    costs = np.full((self.max_disparity), np.inf)

                    for d in range(self.max_disparity):
                        v2 = v1 + d
                        if v2 < l2_cropped:
                            bit_string_1 = bit_strings_1[u, v1]
                            bit_string_2 = bit_strings_2[u, v2]

                            # matching error : hamming distance between the two bistrings
                            cost = np.count_nonzero(bit_string_1 != bit_string_2)

                            costs[d] = cost

                    min_cost = np.min(costs)
                    if min_cost <= self.cost_threshold:
                        disparity = np.argmin(costs)
                        disparity_map[half_block_size + u, half_block_size + v1] = disparity
        elif self.methode == "ssd":
            for u in tqdm(range(h1_cropped)):
                for v1 in range(l1_cropped):
                    costs = np.full((self.max_disparity), np.inf)

                    for d in range(self.max_disparity):
                        v2 = v1 + d
                        if v2 < l2_cropped:
                            bit_string_1 = bit_strings_1[u, v1]
                            bit_string_2 = bit_strings_2[u, v2]

                            # matching error : hamming distance between the two bistrings
                            cost = np.square(bit_string_1 - bit_string_2).sum(axis=2)

                            costs[d] = cost

                    min_cost = np.min(costs)
                    if min_cost <= self.cost_threshold:
                        disparity = np.argmin(costs)
                        disparity_map[half_block_size + u, half_block_size + v1] = disparity 
        elif self.methode == "sad":
            for u in tqdm(range(h1_cropped)):
                for v1 in range(l1_cropped):
                    costs = np.full((self.max_disparity), np.inf)

                    for d in range(self.max_disparity):
                        v2 = v1 + d
                        if v2 < l2_cropped:
                            bit_string_1 = bit_strings_1[u, v1]
                            bit_string_2 = bit_strings_2[u, v2]

                            # matching error : hamming distance between the two bistrings
                            cost = np.sum(np.abs(bit_string_1 - bit_string_2),axis=2)

                            costs[d] = cost

                    min_cost = np.min(costs)
                    if min_cost <= self.cost_threshold:
                        disparity = np.argmin(costs)
                        disparity_map[half_block_size + u, half_block_size + v1] = disparity            
        return disparity_map

    def _faster_matching(self,i_left: np.ndarray,i_right: np.ndarray,) -> np.ndarray:

        (h1, l1) = i_right.shape
        (h2, l2) = i_left.shape

        if h1 != h2:
            raise ValueError("Both images must have the same height")

        # only consider the pixels where the block doesn't go out of the image
        h1_cropped = h1 - self.block_size + 1
        l1_cropped = l1 - self.block_size + 1

        # calculate the bistrings for the first image :
        bit_strings_1 = self._faster_transform(i_right)

        l2_cropped = l2 - self.block_size + 1

        # calculate the bitstrings for the second image
        bit_strings_2 = self._faster_transform(i_left)

        # disparity_map = np.full((h1, l1), np.inf)

        error_map = np.full(
            (h1_cropped, l1_cropped, self.max_disparity + 1),
            np.iinfo(np.uint16).max,
            dtype=np.uint16,
        )

        if self.methode == "census":
            for d in tqdm(range(self.max_disparity + 1)):
                right = bit_strings_1[:, 0 : min(l2_cropped - d, l1_cropped), :]
                left = bit_strings_2[:, d:, :]

                error_map[:, 0 : min(l2_cropped - d, l1_cropped), d] = np.sum(right != left, axis=2)
        elif self.methode == "ssd":
            for d in tqdm(range(self.max_disparity + 1)):
                right = bit_strings_1[:, 0 : min(l2_cropped - d, l1_cropped), :]
                left = bit_strings_2[:, d:, :]

                error_map[:, 0 : min(l2_cropped - d, l1_cropped), d] = np.sum(np.square(right - left), axis=2)
        elif self.methode == "sad":
                for d in tqdm(range(self.max_disparity + 1)):
                    right = bit_strings_1[:, 0 : min(l2_cropped - d, l1_cropped), :]
                    left = bit_strings_2[:, d:, :]
                    error_map[:, 0 : min(l2_cropped - d, l1_cropped), d] = np.sum(np.abs(right - left), axis=2)
        return error_map



if __name__ == "__main__":
    i_left = cv2.imread("./results/rectified_im_l.png", cv2.IMREAD_GRAYSCALE)
    i_right = cv2.imread("./results/rectified_im_r.png", cv2.IMREAD_GRAYSCALE)
    # i_left = cv2.resize(i_left, (i_left.shape[1]//2, i_left.shape[0]//2))
    # i_right = cv2.resize(i_right, (i_right.shape[1]//2, i_right.shape[0]//2))



    matcher = Local_matching(max_disparity=150, block_size=31, seuil_symmetrie=2)
    matcher.optimisation = "python"
    matcher.methode = "ssd"
    a = time.perf_counter()
    disparity = matcher.compute(i_left, i_right)
    b = time.perf_counter()
    print(f"temps de calcul : {b - a} s")

    disparity[disparity==np.inf] = 0

    plt.imshow(disparity, 'gray')
    plt.show()



