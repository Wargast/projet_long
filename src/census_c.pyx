import numpy as np
cimport numpy as np
from cython.parallel import prange
np.import_array()

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef compute_cost(np.uint8_t[:, :, ::1] bit_strings_l, 
                                         np.uint8_t[:, :, ::1] bit_strings_r, 
                                         np.uint16_t[:, :, ::1] error_map,
                                         int u, 
                                         int l1_cropped, 
                                         int l2_cropped,
                                         int max_disparity,
                                         int block_size_squared):
    cdef int cost
    cdef int v1, v2, d, i

    with nogil:
        for v1 in range(l1_cropped):
            for d in range(max_disparity + 1):
                cost = 0
                v2 = v1 + d
                if v2 < l2_cropped:
                    for i in range(block_size_squared):
                        if bit_strings_r[u, v1, i] != bit_strings_l[u, v2, i]:
                            cost += 1
                
                    error_map[u, v1, d] = cost
    


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def census_matching(np.uint8_t[:, :, ::1] bit_strings_l, 
                    np.uint8_t[:, :, ::1] bit_strings_r,
                    int h1_cropped,
                    int l1_cropped,
                    int l2_cropped,
                    int block_size = 3, 
                    int max_disparity = 64):

    cdef int u
    cdef int v1
    cdef int d
    cdef int v2
    cdef int i
    cdef int block_size_squared = block_size * block_size
    cdef np.uint16_t cost

    cdef np.uint16_t[:, :, ::1] error_map = np.full(
        (h1_cropped, l1_cropped, max_disparity + 1),
        np.iinfo(np.uint16).max,
        dtype=np.uint16,
    )

    for u in prange(h1_cropped, nogil=True):
        with gil:
            compute_cost(bit_strings_l, bit_strings_r, error_map, u, l1_cropped, l2_cropped, max_disparity, block_size_squared)
    
    return error_map


