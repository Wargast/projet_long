from dataclasses import dataclass
from pathlib import Path
from re import I
from statistics import pvariance
import sys
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import metrics.census_c as census_c
import cv2
import time as t
import pyvista as pv

from test_recontruction import recontruction, get_Q_from_file
# import pprofile

@dataclass
class Direction:
    """
    represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
    :param direction: (x, y) for cardinal direction.
    :param name: common name of said direction.
    """
    direction: Tuple[int, int]
    name: str



class Paths:
    def __init__(self):
        """
        represent the relation between the directions.
        """
        # 8 defined directions for sgm
        self.N = Direction(direction=(0, -1), name='north')
        self.NE = Direction(direction=(1, -1), name='north-east')
        self.E = Direction(direction=(1, 0), name='east')
        self.SE = Direction(direction=(1, 1), name='south-east')
        self.S = Direction(direction=(0, 1), name='south')
        self.SW = Direction(direction=(-1, 1), name='south-west')
        self.W = Direction(direction=(-1, 0), name='west')
        self.NW = Direction(direction=(-1, -1), name='north-west')
        self.paths = [self.N, self.NE, self.E, self.SE, self.S, self.SW, self.W, self.NW]
        self.size = len(self.paths)
        self.effective_paths = [(self.E,  self.W), (self.SE, self.NW), (self.S, self.N), (self.SW, self.NE)]
 



class SM_matching:
    def __init__(
        self,
        P1: int =10, 
        P2: int =120,
        csize: Tuple[int, int] =(7,7), 
        bsize: Tuple[int, int] =(3,3),
        max_disparity: int =150,
        block_size: int =31,
        cost_threshold: int =150,
        ):
        """__init__ instance de la classe local matching

        Args:
            max_disparity (int, optional): [valeur maximal de matching]. Defaults to 64.
            block_size (int, optional): [taille du bloc de correlation]. Defaults to 9.
            cost_threshold (int, optional): [seuil des erreurs]. Defaults to 150.
            methode (str, optional): [methode utilise (census/ssd/sad)]. Defaults to "census".
            optimisation (str, optional): [optimisation possible (cython/python/python_naif)]. Defaults to "cython".
        """
        self.max_disparity = max_disparity
        self.block_size = block_size
        self.cost_threshold = cost_threshold
        self.P1 = P1
        self.P2 = P2
        self.csize = csize
        self.bsize = bsize
        self.paths = Paths()
                    
    def compute(self,i_left,i_right):
        left_cost_volume, right_cost_volume = census_c.census_matching(
            i_left, i_right, self.block_size, self.max_disparity
        )
        print('\nStarting left aggregation computation...')
        left_aggregation_volume = self.aggregate_costs(left_cost_volume)
        # print('\nStarting right aggregation computation...')
        # right_aggregation_volume = self.aggregate_costs(right_cost_volume)

        print('\nSelecting best disparities...')
        left_disparity_map = np.uint8(self.select_disparity(left_aggregation_volume))
        # right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume), parameters))

        return left_disparity_map
    
    def select_disparity(self, aggregation_volume):
        """
        last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
        :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
        :return: disparity image.
        """
        volume = np.sum(aggregation_volume, axis=3)
        disparity_map = np.argmin(volume, axis=2)
        return disparity_map

    def get_path_cost(self, slice, offset):
        """
        part of the aggregation step, finds the minimum costs in a D x M slice 
        (where M = the number of pixels in the given direction)
        :param slice: M x D array from the cost volume.
        :param offset: ignore the pixels on the border.
        :param parameters: structure containing parameters of the algorithm.
        :return: M x D array of the minimum costs for a given slice in a given 
            direction.
        """
        other_dim = slice.shape[0]
        disparity_dim = slice.shape[1]

        disparities = [d for d in range(disparity_dim)] * disparity_dim
        disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

        penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
        penalties[np.abs(disparities - disparities.T) == 1] = self.P1
        penalties[np.abs(disparities - disparities.T) > 1] = self.P2

        minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
        minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

        for i in range(offset, other_dim):
            previous_cost = minimum_cost_path[i - 1, :]
            current_cost = slice[i, :]
            costs = np.repeat(
                previous_cost, 
                repeats=disparity_dim, 
                axis=0
            ).reshape(disparity_dim, disparity_dim)
            costs = np.amin(costs + penalties, axis=0)
            minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
        return minimum_cost_path

    def get_indices(self, offset, dim, direction, height):
        """
        for the diagonal directions (SE, SW, NW, NE), return the array of 
        indices for the current slice.
        :param offset: difference with the main diagonal of the cost volume.
        :param dim: number of elements along the path.
        :param direction: current aggregation direction.
        :param height: H of the cost volume.
        :return: arrays for the y (H dimension) and x (W dimension) indices.
        """
        y_indices = []
        x_indices = []

        for i in range(0, dim):
            if direction == self.paths.SE.direction:
                if offset < 0:
                    y_indices.append(-offset + i)
                    x_indices.append(0 + i)
                else:
                    y_indices.append(0 + i)
                    x_indices.append(offset + i)

            if direction == self.paths.SW.direction:
                if offset < 0:
                    y_indices.append(height + offset - i)
                    x_indices.append(0 + i)
                else:
                    y_indices.append(height - i)
                    x_indices.append(offset + i)

        return np.array(y_indices), np.array(x_indices)

    def aggregate_costs(self, cost_volume):
        """
        second step of the sgm algorithm, aggregates matching costs for N 
        possible directions (8 in this case).
        :param cost_volume: array containing the matching costs.
        :param parameters: structure containing parameters of the algorithm.
        :param paths: structure containing all directions in which to 
            aggregate costs.
        :return: H x W x D x N array of matching cost for all defined 
            directions.
        """
        height = cost_volume.shape[0]
        width = cost_volume.shape[1]
        disparities = cost_volume.shape[2]
        start = -(height - 1)
        end = width - 1

        aggregation_volume = np.zeros(
            shape=(height, width, disparities, self.paths.size), 
            dtype=cost_volume.dtype
        )

        path_id = 0
        main_aggregation = np.zeros(
            shape=(height, width, disparities), 
            dtype=cost_volume.dtype
        )
        opposite_aggregation = np.copy(main_aggregation)
        for path in self.paths.effective_paths:
            print('\tProcessing paths {} and {}...'.format(
                path[0].name, path[1].name), end='\n')
            sys.stdout.flush()
            dawn = t.time()

            main_aggregation.fill(0)
            opposite_aggregation.fill(0)

            main = path[0]
            if main.direction == self.paths.S.direction:
                for x in tqdm(range(0, width)):
                    south = cost_volume[0:height, x, :]
                    north = np.flip(south, axis=0)
                    main_aggregation[:, x, :] = self.get_path_cost(south, 1)
                    opposite_aggregation[:, x, :] = np.flip(
                        self.get_path_cost(north, 1), axis=0)

            if main.direction == self.paths.E.direction:
                for y in tqdm(range(0, height)):
                    east = cost_volume[y, 0:width, :]
                    west = np.flip(east, axis=0)
                    main_aggregation[y, :, :] = self.get_path_cost(east, 1)
                    opposite_aggregation[y, :, :] = np.flip(
                        self.get_path_cost(west, 1), axis=0)

            if main.direction == self.paths.SE.direction:
                for offset in tqdm(range(start, end)):
                    south_east = cost_volume.diagonal(offset=offset).T
                    north_west = np.flip(south_east, axis=0)
                    dim = south_east.shape[0]
                    y_se_idx, x_se_idx = self.get_indices(
                        offset, 
                        dim, 
                        self.paths.SE.direction, None
                    )
                    y_nw_idx = np.flip(y_se_idx, axis=0)
                    x_nw_idx = np.flip(x_se_idx, axis=0)
                    main_aggregation[y_se_idx, x_se_idx, :] = self.get_path_cost(south_east, 1)
                    opposite_aggregation[y_nw_idx, x_nw_idx, :] = self.get_path_cost(north_west, 1)

            if main.direction == self.paths.SW.direction:
                for offset in tqdm(range(start, end)):
                    south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                    north_east = np.flip(south_west, axis=0)
                    dim = south_west.shape[0]
                    y_sw_idx, x_sw_idx = self.get_indices(offset, dim, self.paths.SW.direction, height - 1)
                    y_ne_idx = np.flip(y_sw_idx, axis=0)
                    x_ne_idx = np.flip(x_sw_idx, axis=0)
                    main_aggregation[y_sw_idx, x_sw_idx, :] = self.get_path_cost(south_west, 1)
                    opposite_aggregation[y_ne_idx, x_ne_idx, :] = self.get_path_cost(north_east, 1)

            print("start fill agregation_volume")
            aggregation_volume[:, :, :, path_id] = main_aggregation
            aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
            path_id = path_id + 2

            dusk = t.time()
            print('\t(done in {:.2f}s)'.format(dusk - dawn))

        return aggregation_volume


if __name__ == "__main__":
    dataset_path = Path("datas/middlebury/artroom1")
    i_left = cv2.imread((dataset_path/"im0.png").as_posix(), cv2.IMREAD_GRAYSCALE)
    i_right = cv2.imread((dataset_path/"im1.png").as_posix(), cv2.IMREAD_GRAYSCALE)
    i_left = cv2.resize(i_left, (i_left.shape[1]//2, i_left.shape[0]//2))
    i_right = cv2.resize(i_right, (i_right.shape[1]//2, i_right.shape[0]//2))

    matcher = SM_matching()
    a = t.perf_counter()
    disparity = matcher.compute(i_left,i_right)
    b = t.perf_counter()
    print(f"temps de calcul : {b - a} s")

    disparity[disparity==np.inf] = 0

    plt.imshow(disparity, 'gray')
    plt.show()

    Q = get_Q_from_file(file=dataset_path/'calib.txt')
    points_p = recontruction(disparity, Q)
    point_cloud = pv.PolyData(points_p)
    point_cloud.plot(eye_dome_lighting=True, point_size=2.0)

