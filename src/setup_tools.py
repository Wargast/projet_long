import pickle
from pprint import pprint
import cv2
import numpy as np
from pathlib import Path


def parse_calib_file(path: Path):
    """
    patern:
        cam1=[fx 0 cx; 0 fy cy; 0 0 1]
        doffs: int
        baseline: float
        width:int
        height:int
        ndisp:int
        vmin:int
        vmax:int
    """
    params = {}
    with open(path, 'r') as file:
        while True:
            line = file.readline().strip()
            if not line:
                break

            lenght_name = line.find("=")
            var_name = line[:lenght_name]
            if line[lenght_name+1] == '[':
                ll = line[lenght_name+2:-1].split(";")
                value = np.array([list(map(float, l.strip().split(" "))) for l in ll])
            else:
                value = float(line[lenght_name+1:])
            params[var_name] = value
    return params

def parse_pickle(path:Path):
    param = {}
    with open(path, 'rb') as file:
        # Call load method to deserialze
        param = pickle.load(file)
    return param
  
def stereoBM_from_file(fname):
    stereo_params = parse_pickle(fname)
    pprint(stereo_params)

    stereo = cv2.StereoBM_create()
    stereo.setNumDisparities(stereo_params["numDisparities"])
    stereo.setBlockSize(stereo_params["blockSize"])
    stereo.setPreFilterType(stereo_params["preFilterType"])
    stereo.setPreFilterSize(stereo_params["preFilterSize"])
    stereo.setPreFilterCap(stereo_params["preFilterCap"])
    stereo.setTextureThreshold(stereo_params["textureThreshold"])
    stereo.setUniquenessRatio(stereo_params["uniquenessRatio"])
    stereo.setSpeckleRange(stereo_params["speckleRange"])
    stereo.setSpeckleWindowSize(stereo_params["speckleWindowSize"])
    stereo.setDisp12MaxDiff(stereo_params["disp12MaxDiff"])
    stereo.setMinDisparity(stereo_params["minDisparity"])
    return stereo

def stereoSGBM_from_file(fname):
    stereo_params = parse_pickle(fname)
    pprint(stereo_params)

    stereo = cv2.StereoSGBM_create()
    stereo.setNumDisparities(stereo_params["numDisparities"])
    stereo.setBlockSize(stereo_params["blockSize"])
    stereo.setUniquenessRatio(stereo_params["uniquenessRatio"])
    stereo.setSpeckleRange(stereo_params["speckleRange"])
    stereo.setSpeckleWindowSize(stereo_params["speckleWindowSize"])
    stereo.setDisp12MaxDiff(stereo_params["disp12MaxDiff"])
    stereo.setMinDisparity(stereo_params["minDisparity"])

    return stereo



if __name__ == "__main__":
    calib_file = Path("datas/middlebury/ladder1/calib.txt")
    parse_calib_file(calib_file)