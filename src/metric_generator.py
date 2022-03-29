import time
from matplotlib import pyplot as plt
from pypfm import PFMLoader
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from local_matching import Local_matching

def rms(I_diff):
    # make mean squared error    
    error = np.sqrt(np.square(I_diff))
    error = error.mean(axis=None)
    return error

def bad(I_diff: np.ndarray, alpha: float):
    bad_pix = I_diff[I_diff>alpha]
    return bad_pix.size/I_diff.size

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
        "bad50" : (bad, 50),
        "bad20" : (bad, 20),
        "bad10" : (bad, 10),
        "bad2.0": (bad, 2.0),
        "bad1.0": (bad, 1.0),
        "bad0.5": (bad, 0.5),
    }

    result_metrics = []
    df = pd.DataFrame( 
        columns=["data", "execution_time", "covering"] + list(f_error.keys())
    )
    # print(df)
    # stereo = stereoBM_from_file("results/param.pkl")
    # stereo.setMinDisparity(0)
    method="ssd"
    max_disparity=150
    block_size=9
    seuil_symmetrie=0
    stereo = Local_matching(
        max_disparity=max_disparity,
        block_size=block_size, 
        seuil_symmetrie=seuil_symmetrie,
        method=method
    )
    
    # iter on dataset
    i = 0
    for data in tqdm(datas_to_process):
        i += 1
        if i > 5:
            break
        # open image 
        print(data.name.ljust(30))
        img0 = cv2.imread((data/'im0.png').as_posix(), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread((data/'im1.png').as_posix(), cv2.IMREAD_GRAYSCALE)

        # open GS disp_map
        map_ref0 = loader.load_pfm(data/'disp0.pfm')
        map_ref1 = loader.load_pfm(data/'disp1.pfm')

        start = time.time()
        # process each algo on stereocouple
        disparity_map = stereo.compute(img0, img1)
        end_time = time.time()
        exec_time = end_time - start
        
        df.loc[len(df.index)] = 0

        df.iloc[len(df.index)-1, 0] = data.name
        df.iloc[len(df.index)-1, 1] = exec_time
        
        # make diff and filter non revelant values
        I_filtred = disparity_map.copy()
        I_filtred[I_filtred<=0] = np.inf 
        I_diff = map_ref0 - I_filtred
        I_diff = I_diff[~np.isnan(I_diff)]
        I_diff = I_diff[~np.isinf(I_diff)]   
        df.iloc[len(df.index)-1, 2] = I_diff.size/disparity_map.size

        # append each metrics
        for f_name, [f, param] in f_error.items():
            # print(f"process metric {f_name}...")
            if param is not None:
                res = float(f(I_diff, param))
            else:
                res = float(f(I_diff))

            df.loc[len(df.index)-1, f_name] = res
        print(df)

        # plot_res(data.name, disparity_map, map_ref0, img0 )

        # del cost_map
        del disparity_map
        result_metrics.append(df)
        # break
        
    print(df)

    # generate stat on result
    df.to_csv(f"results/census-disp={max_disparity}-"+
                        f"block_size={block_size}-"+
                        f"seuil_symmetrie={seuil_symmetrie}.csv")


def plot_res(name, disparity_map, map_ref, img ):
    plt.figure(name)#, figsize=(160,90))
    plt.subplot(2,2,1)
    disparity_map[np.isinf(disparity_map)] = 0
    plt.imshow(disparity_map, 'gray')
    plt.title("disparity map 0")
    plt.subplot(2,2,2)
    plt.imshow(map_ref, 'gray')
    plt.title("Goal Stantard disparity map 0")

    errormap_to_show = disparity_map - map_ref
    errormap_to_show[np.isnan(errormap_to_show)] = 0
    errormap_to_show[np.isinf(errormap_to_show)] = 0

    plt.subplot(2,2,3)
    plt.imshow(np.abs(errormap_to_show), 'gray')
    plt.title("error (|GS - disp_map|)")
    plt.subplot(2,2,4)
    plt.imshow(img, "gray")
    plt.title("image origine")

    plt.show()


if __name__ == "__main__":
    main()