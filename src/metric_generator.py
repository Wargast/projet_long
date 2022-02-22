from cmath import inf
from distutils.log import error
from sys import exec_prefix
import time
from unittest import result
from pypfm import PFMLoader
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from setup_tools import stereoBM_from_file, stereoSGBM_from_file

# import algo of disparity map calculation
from census import faster_census_matching, naive_census_matching, disparity_from_error_map

def rms(I_ref, I):
    # make mean squared error on echea res in df

    error = np.square(I - I_ref)

    # filter nan and inf
    error = error[~np.isnan(error)]
    error = error[~np.isinf(error)]
    error = error.mean(axis=None)
    return error

def bad(I_ref: np.ndarray, I: np.ndarray, alpha: float):
    diff = I - I_ref
    bad_pix = diff[diff>alpha]
    return bad_pix.size/I_ref.size


def main():
    loader = PFMLoader(color=False, compress=False)
    # get all data to test and iter on them
    dataset_path = Path("datas/middlebury/")
    datas_to_process = [
        data_file for data_file in dataset_path.iterdir() 
        if data_file.is_dir()
    ]
    print(f"dataset size: {len(datas_to_process)}")
    
    # dict of function to test name as key
    f_error = {
        "rms": (rms, None),
        "bad50": (bad, 50),
        "bad200": (bad, 200),
        "bad100": (bad,100),
        "bad300": (bad, 300),
        "bad20": (bad, 20),
        "bad2.0": (bad, 2.0),
        "bad1.0": (bad, 1.0),
        "bad0.5": (bad, 0.5),
    }

    result_metrics = []
    df = pd.DataFrame( 
        columns=["data", "execution_time"] + list(f_error.keys())
    )
    print(df)
    stereo = stereoBM_from_file("results/param.pkl")
    # iter on dataset
    for data in tqdm(datas_to_process):
        # open image 
        print(data.name.ljust(30))
        img0 = cv2.imread((data/'im0.png').as_posix(), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread((data/'im1.png').as_posix(), cv2.IMREAD_GRAYSCALE)

        # open GS disp_map
        map_ref = loader.load_pfm(data/'disp0.pfm')

        # check if GS is transposed 
        if map_ref.shape != img0.shape[:2]:
            map_ref = map_ref.transpose()

        start = time.time()
        # process each algo on stereocouple
        disparity_map = stereo.compute(img0, img1)/16
        end_time = time.time()
        exec_time = end_time - start
        
        df.loc[len(df.index)] = 0
        print(df)

        df.iloc[len(df.index)-1, 0] = data.name
        df.iloc[len(df.index)-1, 1] = exec_time

        for f_name, [f, param] in f_error.items():
            print(f"process metric {f_name}...")
            if param is not None:
                res = f(disparity_map, map_ref, param)
            else:
                res = f(disparity_map, map_ref)

            df.loc[len(df.index)-1, f_name] = res
            


        # del cost_map
        del disparity_map
        result_metrics.append(df)
        
    print(df)
    # generate stat on result
    df.to_csv(f"results/opencv_version.csv")





if __name__ == "__main__":
    main()