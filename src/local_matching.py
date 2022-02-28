from re import I
import numpy as np
from tqdm import tqdm
import census_c
import cv2
import time
# import pprofile

class Local_matching:
    def __init__(
        self,
        max_disparity: int =64,
        block_size: int = 9,
        cost_threshold: int = 150,
        methode: str = "census",
        ):
        """__init__ instance de la classe local matching

        Args:
            max_disparity (int, optional): [valeur maximal de matching]. Defaults to 64.
            block_size (int, optional): [taille du bloc de correlation]. Defaults to 9.
            cost_threshold (int, optional): [seuil des erreurs]. Defaults to 150.
            methode (str, optional): [methode utilise (census/ssd/sad)]. Defaults to "census".
            optimisation (str, optional): [optimisation possible (cython/python/python_naif)]. Defaults to "cython".
        """
        self.methode = methode
        self.max_disparity = max_disparity
        self.block_size = block_size
        self.cost_threshold = cost_threshold
        if methode == "census":
            self.cost_threshold = 150
        elif methode == "ssd" or methode == "sad":
            self.cost_threshold = 1000
                    
    def compute(self,i_left,i_right):
        error_map_l, error_map_r = census_c.census_matching(
            i_left, i_right, self.block_size, self.max_disparity
        )
            
        return self._disparity_from_error_map(error_map_l, error_map_r)[0]
    

    def _disparity_from_error_map(self,
    error_map_l: np.ndarray, 
    error_map_r: np.ndarray, 
    ) -> np.ndarray:

        half_block_size = self.block_size // 2

        disparity_map_left = np.argmin(error_map_l, axis=2).astype(np.float32)
        disparity_map_right = np.argmin(error_map_r, axis=2).astype(np.float32)

        (h, l, _) = error_map_l.shape

        ind = np.meshgrid(range(h), range(l), indexing='ij')

        ind2 = (ind[0], (ind[1] - disparity_map_left).astype(np.uint16))

        d2 = disparity_map_right[ind2]

        disparity_map_left_filtered = disparity_map_left.copy()

        disparity_map_left_filtered[d2 != disparity_map_left] = np.inf

        disparity_map_left_filtered = np.pad(
            disparity_map_left_filtered,
            (half_block_size, half_block_size),
            mode="constant",
            constant_values=np.inf,
        )

        disparity_map_left = np.pad(
            disparity_map_left,
            (half_block_size, half_block_size),
            mode="constant",
            constant_values=np.inf,
        )

        return disparity_map_left_filtered, disparity_map_left


if __name__ == "__main__":
    i_left = cv2.imread("./datas/middlebury/artroom1/im0.png", cv2.IMREAD_GRAYSCALE)
    i_right = cv2.imread("./datas/middlebury/artroom1/im1.png", cv2.IMREAD_GRAYSCALE)

    matcher = Local_matching()
    error_map = matcher.compute(i_left,i_right)
    matcher.optimisation = "python"
    matcher.methode = "ssd"
    error_map2 = matcher.compute(i_left,i_right)
    matcher.optimisation = "python_naif"
    # error_map3 = matcher.compute(i_left,i_right)
    # print(np.array_equal(error_map, error_map2))

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
