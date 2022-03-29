import numpy as np
cimport numpy as np
from cython.parallel import prange
np.import_array()

cimport cython
from numpy.math cimport INFINITY

import time

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void aux_func(np.uint8_t[:, ::1] im_left, 
                   np.uint8_t[:, ::1] im_right, 
                   np.uint32_t[:, :, ::1] error_map_l, 
                   np.uint32_t[:, :, ::1] error_map_r, 
                   np.uint16_t[:, :, ::1] squared_diffs,
                   np.uint32_t[:, :, ::1] first_pass_res,
                   int h1, 
                   int l1, 
                   int d, 
                   int block_size) nogil:

    cdef int u, v
    cdef int diff

    for u in range(h1):
        for v in range(l1 - d):
            diff = im_right[u, v] - im_left[u, v + d]
            squared_diffs[d, u, v] = diff * diff

    convolve(squared_diffs, first_pass_res, error_map_l, error_map_r, block_size, d)
    

@cython.boundscheck(False)
@cython.wraparound(False)
def ssd_sad(np.uint8_t[:, ::1] im_left,
            np.uint8_t[:, ::1] im_right,
            int block_size, 
            int max_disparity):

    cdef int h1 = im_right.shape[0]
    cdef int l1 = im_right.shape[1]
    cdef int h2 = im_left.shape[0]
    cdef int l2 = im_left.shape[1]

    if h1 != h2:
        raise ValueError("Both images must have the same height")

    if block_size % 2 == 0:
        raise ValueError("Block size must be an odd number")

    cdef int h1_cropped = h1 - block_size + 1
    cdef int l1_cropped = l1 - block_size + 1
    cdef int l2_cropped = l2 - block_size + 1

    cdef np.uint32_t[:, :, ::1] error_map_l = np.full(
        (h1_cropped, l2_cropped, max_disparity + 1),
        4294967295, #max cost value (32 bits)
        dtype=np.uint32,
    )

    cdef np.uint32_t[:, :, ::1] error_map_r = np.full(
        (h1_cropped, l1_cropped, max_disparity + 1),
        4294967295, #max cost value (32 bits)
        dtype=np.uint32,
    )

    cdef np.uint16_t[:, :, ::1] squared_diffs = np.zeros(
        (max_disparity + 1, h1, l1),
        dtype=np.uint16,
    )

    cdef np.uint32_t[:, :, ::1] first_pass_res = np.zeros(
        (max_disparity + 1, h1, l1_cropped),
        dtype=np.uint32,
    )

    cdef int d

    a = time.perf_counter()

    # for d in prange(max_disparity + 1, nogil=True, schedule='static'):
    for d in range(max_disparity + 1):
        aux_func(im_left,
                im_right,
                error_map_l,
                error_map_r,
                squared_diffs,
                first_pass_res,
                h1,
                l1,
                d,
                block_size)     

    b = time.perf_counter()
           
    print(f"temps : {b - a} s")

    return error_map_l.base, error_map_r.base


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void convolve(np.uint16_t[:, :, ::1] squared_diffs, 
                   np.uint32_t[:, :, ::1] first_pass_res, 
                   np.uint32_t[:, :, ::1] error_map_l, 
                   np.uint32_t[:, :, ::1] error_map_r, 
                   int block_size, 
                   int d) nogil:

    cdef int h = squared_diffs.shape[1]
    cdef int l = squared_diffs.shape[2]

    cdef int half_block_size = block_size // 2

    cdef int h_cropped = h - block_size + 1
    cdef int l_cropped = l - block_size + 1

    cdef int sum
    cdef int u, v

    for u in range(h):
        sum = 0
        for v in range(block_size):
            sum += squared_diffs[d, u, v]
        first_pass_res[d, u, 0] = sum
        
        for v in range(1, l_cropped - d):
            sum = sum - squared_diffs[d, u, v - 1] + squared_diffs[d, u, v + block_size - 1]
            first_pass_res[d, u, v] = sum

    # cdef np.uint32_t[::1, :] first_pass_res_t = np.asfortranarray(first_pass_res)

    for v in range(l_cropped - d):
        sum = 0
        for u in range(block_size):
            sum += first_pass_res[d, u, v]
        error_map_r[0, v, d] = sum
        error_map_l[0, v + d, d] = sum

        for u in range(1, h_cropped):
            sum = sum - first_pass_res[d, u - 1, v] + first_pass_res[d, u + block_size - 1, v]
            error_map_r[u, v, d] = sum
            error_map_l[u, v + d, d] = sum




@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_line_disparity(np.uint32_t[:, :, ::1] error_map,
                                 np.int16_t[:, ::1] disparity_map,
                                 int u,
                                 int num_disparities,
                                 int l_cropped) nogil:
    cdef int v
    cdef int d
    cdef int min_cost
    cdef int min_index
    for v in range(l_cropped):
        min_cost = 65535
        best_disparity = -1
        for d in range(num_disparities):
            if error_map[u, v, d] < min_cost:
                min_cost = error_map[u, v, d]
                best_disparity = d
        disparity_map[u, v] = best_disparity


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void check_line_symmetry(np.int16_t[:, ::1] disparity_map_l,
                              np.int16_t[:, ::1] disparity_map_r,
                              np.float32_t[:, ::1] disparity_map_l_filtered,
                              int u,
                              int l1_cropped,
                              int half_block_size,
                              int seuil_symmetrie) nogil:
    cdef int v1, v2
    for v1 in range(l1_cropped):
        v2 = v1 - disparity_map_l[u, v1]
        if (disparity_map_l[u, v1] - disparity_map_r[u, v2]) <= seuil_symmetrie and (disparity_map_l[u, v1] - disparity_map_r[u, v2]) >= -seuil_symmetrie:
            disparity_map_l_filtered[u + half_block_size, v1 + half_block_size] = disparity_map_l[u, v1]


@cython.boundscheck(False)
@cython.wraparound(False)
def disparity_from_error_map(np.uint32_t[:, :, ::1] error_map_l,
                             np.uint32_t[:, :, ::1] error_map_r,
                             int block_size,
                             int seuil_symmetrie=0):

    cdef int h1_cropped = error_map_l.shape[0]
    cdef int l1_cropped = error_map_l.shape[1]
    cdef int l2_cropped = error_map_r.shape[1]
    cdef int num_disparities = error_map_l.shape[2]

    cdef int h1 = h1_cropped - 1 + block_size
    cdef int l1 = l1_cropped - 1 + block_size

    cdef int half_block_size = block_size // 2
    
    cdef np.int16_t[:, ::1] disparity_map_l = np.zeros(
        (h1_cropped, l1_cropped),
        dtype=np.int16
    )
    
    cdef np.int16_t[:, ::1] disparity_map_r = np.zeros(
        (h1_cropped, l2_cropped),
        dtype=np.int16
    )

    cdef np.float32_t[:, ::1] disparity_map_l_filtered = np.full(
        (h1, l1),
        INFINITY,
        dtype=np.float32
    )

    cdef int u

    for u in prange(h1_cropped, nogil=True, schedule='static'):
        compute_line_disparity(error_map_l, disparity_map_l, u, num_disparities, l1_cropped)

    for u in prange(h1_cropped, nogil=True, schedule='static'):
        compute_line_disparity(error_map_r, disparity_map_r, u, num_disparities, l2_cropped)

    for u in prange(h1_cropped, nogil=True, schedule='static'):
        check_line_symmetry(disparity_map_l, disparity_map_r, disparity_map_l_filtered, u, l1_cropped, half_block_size, seuil_symmetrie)

    return disparity_map_l_filtered.base