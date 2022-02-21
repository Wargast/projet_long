import numpy as np
import cv2

def sad(I1,I2,d):
    """AI is creating summary for sad

    Args:
        I1 ([type]): [description]
        I2 ([type]): [description]
        d ([type]): [description]

    Returns:
        [type]: [description]
    """
    Res = np.zeros([I1.shape[0],I1.shape[1],I1.shape[1]])
    for i in range(1,I1.shape[0]):
        for j in range(1,I1.shape[1]):
            for d in range(I1.shape[0]):
                block_error = 0
                for u in [-1,0,1]:
                    for v in [-1,0,1]:
                        block_error = block_error + I1(i+u,j+v)-I2(i+u+d,j+v)
                Res[i,j,d]=block_error
    return Res