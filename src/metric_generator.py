from cmath import inf
from sys import exec_prefix
import time
from unittest import result
from pypfm import PFMLoader
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# import algo of disparity map calculation
def foo(img0, img1):
    # fake disparity map calculation 
    # only for test
    return [ np.random.randint(0,255, size=(img0.shape[0], img0.shape[1], img0.shape[0]), dtype='uint8' ) ]*2

def disparity_map_from_cost_matrix(cost_mat: np.ndarray):
    ind_min = np.argmin(cost_mat, axis=2)
    res = np.arange(0, cost_mat.shape[1], 1)
    res = np.expand_dims(res, 0)
    res = np.repeat(res, cost_mat.shape[0],axis=0)

    res = np.abs(res - ind_min)
    return res

def main():
    loader = PFMLoader(color=False, compress=False)
    # get all data to test and iter on them
    dataset_path = Path("datas/middlebury/")
    datas_to_process = [
        data_file for data_file in dataset_path.iterdir() 
        if data_file.is_dir()
    ]
    print(f"dataset size: {len(datas_to_process)}")
    
    f_to_test = {
        "foo": foo,
        "fooBis": foo
    }

    result_metrics = []
    df = pd.DataFrame( columns=['function', "data", "execution_time", "score_0", "score_1"])
    print(df)
    # iter on data
    for data in datas_to_process:
        # open image 
        print(data.name.ljust(30))
        img0 = cv2.imread((data/'im0.png').as_posix())
        img1 = cv2.imread((data/'im1.png').as_posix())

        # open GS disp_map
        pfm0 = loader.load_pfm(data/'disp0.pfm')
        pfm1 = loader.load_pfm(data/'disp1.pfm')

        # check if GS is transposed 
        if pfm0.shape != img0.shape[:2]:
            pfm0 = pfm0.transpose()
            pfm1 = pfm1.transpose()

        # process each algo on stereocouple
        
        for f_name, f in f_to_test.items():
            start = time.time()
            cost_mat0, cost_mat1 = f(img0, img1)
            cost_time = time.time()
            disp_map0 = disparity_map_from_cost_matrix(cost_mat0)
            # disp_map1 = disparity_map_from_cost_matrix(cost_mat1)
            disp_map1 = np.zeros(disp_map0.shape)


            end_time = time.time()
            exec_time = end_time - start
            # make mean squared error on echea res in df
            error0 = np.square(disp_map0 - pfm0).mean(axis=None)
            error1 = np.square(disp_map1 - pfm1).mean(axis=None)

            df.loc[len(df.index)] = [f_name, data.name, exec_time, error0, error1]

            print("\t-> ", exec_time, cost_time-start, end_time-cost_time, error0, error1)

            del cost_mat0
            del cost_mat1
            del disp_map0
            del disp_map1
        result_metrics.append(df)
        pass
    print(df)
    # generate stat on result





if __name__ == "__main__":
    main()