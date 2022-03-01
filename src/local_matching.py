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
        seuil_symmetrie: int = 0
    ):
        """__init__ instance de la classe local matching

        Args:
            max_disparity (int, optional): [valeur maximal de matching]. Defaults to 64.
            block_size (int, optional): [taille du bloc de correlation]. Defaults to 9.
            methode (str, optional): [methode utilisée (census/ssd/sad)]. Defaults to "census".
        """
        self.method = method
        self.max_disparity = max_disparity
        self.block_size = block_size
        self.seuil_symmetrie = seuil_symmetrie


    def compute(self, i_left: np.ndarray, i_right: np.ndarray):
        if self.method == "census":
            error_map_l, error_map_r = census_c.census_matching(
                i_left, i_right, self.block_size, self.max_disparity
            )

        else:
            raise ValueError(f"méthode {self.method} non existante")

        return self._disparity_from_error_map(error_map_l, error_map_r)


    def _disparity_from_error_map(
        self,
        error_map_l: np.ndarray,
        error_map_r: np.ndarray,
    ) -> np.ndarray:

        return census_c.disparity_from_error_map(
            error_map_l, error_map_r, self.block_size, self.seuil_symmetrie
        )


if __name__ == "__main__":
    i_left = cv2.imread("./datas/middlebury/artroom1/im0.png", cv2.IMREAD_GRAYSCALE)
    i_right = cv2.imread("./datas/middlebury/artroom1/im1.png", cv2.IMREAD_GRAYSCALE)

    matcher = Local_matching(max_disparity=150, block_size=11, method="census", seuil_symmetrie=10)
    a = time.perf_counter()
    disparity = matcher.compute(i_left, i_right)
    b = time.perf_counter()
    print(f"temps de calcul : {b - a} s")

    disparity[disparity==np.inf] = 0

    plt.imshow(disparity, 'gray')
    plt.show()



