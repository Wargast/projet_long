import numpy as np
from tqdm import tqdm
cimport numpy as np
np.import_array()

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def census_matching(np.ndarray[np.uint8_t, ndim=3] bit_strings_l, 
                    np.ndarray[np.uint8_t, ndim=3] bit_strings_r,
                    int h1_cropped,
                    int l1_cropped,
                    int l2_cropped,
                    int block_size = 3, 
                    int max_disparity = 64):

    cdef int u
    cdef int v1
    cdef int d
    cdef int v2
    cdef np.uint16_t cost
    cdef np.ndarray[np.uint8_t, ndim=1] bsr
    cdef np.ndarray[np.uint8_t, ndim=1] bsl

    cdef np.ndarray[np.uint16_t, ndim=3] error_map = np.full(
        (h1_cropped, l1_cropped, max_disparity + 1),
        np.iinfo(np.uint16).max,
        dtype=np.uint16,
    )

    for u in tqdm(range(h1_cropped)):
        for v1 in range(l1_cropped):
            for d in range(max_disparity + 1):
                cost = 0
                v2 = v1 + d
                if v2 < l2_cropped:
                    for i in range(block_size**2):
                        if bit_strings_r[u, v1, i] != bit_strings_l[u, v2, i]:
                            cost += 1
                
                    error_map[u, v1, d] = cost
    
    return error_map


