from cmath import inf
from sys import exec_prefix
import time
from unittest import result
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from pfm import read_pfm

# import algo of disparity map calculation
from census import faster_census_matching, naive_census_matching, disparity_from_error_map

def main():
    # get all data to test and iter on them
    dataset_path = Path("datas/middlebury/")
    datas_to_process = [
        data_file for data_file in dataset_path.iterdir() 
        if data_file.is_dir()
    ]
    print(f"dataset size: {len(datas_to_process)}")
    
    # dict of function to test name as key
    f_to_test = {
        # "naive_census_matching": naive_census_matching,
        "faster_census_matching": faster_census_matching,
    }

    result_metrics = []
    df = pd.DataFrame( columns=['function', "data", "execution_time", "score_0"])

    # iter on dataset
    for data in datas_to_process:
        # open image 
        print(data.name.ljust(30))
        img0 = cv2.imread((data/'im0.png').as_posix(), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread((data/'im1.png').as_posix(), cv2.IMREAD_GRAYSCALE)

        # open GS disp_map
        pfm0 = read_pfm(data/'disp0.pfm')

        # check if GS is transposed 
        if pfm0.shape != img0.shape[:2]:
            pfm0 = pfm0.transpose()

        # process each algo on stereocouple
        
        for f_name, f in f_to_test.items():
            print(f"process metric for {f_name}...")
            start = time.time()
            cost_map = f(img0, img1)
            disparity_map = disparity_from_error_map(
                cost_map, block_size=15, cost_threshold=100
            )

            end_time = time.time()
            exec_time = end_time - start


            # make mean squared error on echea res in df

            error = np.square(disparity_map - pfm0)
            # filter nan and inf
            error = error[~np.isnan(error)]
            error = error[~np.isinf(error)]
            error = error.mean(axis=None)

            df.loc[len(df.index)] = [f_name, data.name, exec_time, error]

            print("\t-> ", exec_time, error)

            del cost_map
            del disparity_map
        result_metrics.append(df)
        
    print(df)
    # generate stat on result
    df.to_csv(f"results/metrics/{'-'.join(f_to_test.keys())}.csv")





if __name__ == "__main__":
    main()