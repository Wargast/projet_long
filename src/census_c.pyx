import numpy as np
cimport numpy as np
from cython.parallel import prange
np.import_array()

cimport cython


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline void line_census_transform(np.uint64_t[:, ::1] bit_strings, 
                                       np.uint8_t[:, ::1] im, 
                                       int u,
                                       int block_size, 
                                       int l_cropped,
                                       int half_block_size) nogil:

    cdef int v
    cdef int i, j
    cdef int center_value
    cdef np.uint64_t bit_str

    for v in range(l_cropped):
        bit_str = 0
        center_value = im[u + half_block_size, v + half_block_size]
        for i in range(block_size):
            for j in range(block_size):
                bit_str = (bit_str << 1) | (im[u + i, v + j] < center_value)
        bit_strings[u, v] = bit_str


@cython.boundscheck(False) 
@cython.wraparound(False)  
cdef np.uint64_t[:, ::1] census_transform(np.uint8_t[:, ::1] im, int block_size):

    cdef int h = im.shape[0]
    cdef int l = im.shape[1]
    cdef int h_cropped = h - block_size + 1
    cdef int l_cropped = l - block_size + 1

    cdef int half_block_size = block_size // 2

    cdef np.uint64_t[:, ::1] bit_strings = np.zeros((h_cropped, l_cropped), dtype=np.uint64)
    
    cdef int u

    for u in prange(h_cropped, nogil=True, schedule='static'):
        line_census_transform(bit_strings, im, u, block_size, l_cropped, half_block_size)
                    
    return bit_strings


cdef inline int hamming_distance(np.uint64_t a, np.uint64_t b) nogil:
    cdef np.uint64_t x = a^b
    cdef int c = 0
    while x != 0:
        x &= x-1
        c += 1
    return c


cdef inline int faster_hamming_distance(np.uint64_t a, np.uint64_t b) nogil:
    cdef np.uint64_t x = a^b

    #SWAR popcount
    cdef np.uint64_t k1 = 6148914691236517205 #0x5555555555555555
    cdef np.uint64_t k2 = 3689348814741910323 #0x3333333333333333
    cdef np.uint64_t k4 = 1085102592571150095 #0x0f0f0f0f0f0f0f0f
    cdef np.uint64_t kf = 72340172838076673 #0x0101010101010101

    x = x - ((x >> 1) & k1) #put count of each 2 bits into those 2 bits
    x = (x & k2) + ((x >> 2) & k2) #put count of each 4 bits into those 4 bits
    x = (x + (x >> 4)) & k4 #put count of each 8 bits into those 8 bits
    x = (x * kf) >> 56 #returns 8 most significant bits of x + (x<<8) + (x<<16) + (x<<24) + ...
    
    return x;


@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef inline void compute_line_costs(np.uint64_t[:, ::1] bit_strings_l, 
                                    np.uint64_t[:, ::1] bit_strings_r, 
                                    np.uint16_t[:, :, ::1] error_map_l,
                                    np.uint16_t[:, :, ::1] error_map_r,
                                    int u, 
                                    int l1_cropped, 
                                    int l2_cropped,
                                    int max_disparity) nogil:

    cdef int cost
    cdef int v1, v2, d, i

    for v1 in range(l1_cropped):
        for d in range(max_disparity + 1):
            cost = 0
            v2 = v1 + d
            if v2 < l2_cropped:
                cost = faster_hamming_distance(bit_strings_r[u, v1], bit_strings_l[u, v2])
                error_map_r[u, v1, d] = cost
                error_map_l[u, v2, d] = cost
    

@cython.boundscheck(False)
@cython.wraparound(False)
def census_matching(np.uint8_t[:, ::1] im_left,
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

    cdef np.uint64_t[:, ::1] bit_strings_l = census_transform(im_left, block_size)
    cdef np.uint64_t[:, ::1] bit_strings_r = census_transform(im_right, block_size)

    cdef np.uint16_t[:, :, ::1] error_map_l = np.full(
        (h1_cropped, l1_cropped, max_disparity + 1),
        np.iinfo(np.uint16).max,
        dtype=np.uint16,
    )

    cdef np.uint16_t[:, :, ::1] error_map_r = np.full(
        (h1_cropped, l2_cropped, max_disparity + 1),
        np.iinfo(np.uint16).max,
        dtype=np.uint16,
    )

    cdef int u

    for u in prange(h1_cropped, nogil=True, schedule='static'):
        compute_line_costs(bit_strings_l, bit_strings_r, error_map_l, error_map_r, u, l1_cropped, l2_cropped, max_disparity)
    
    return error_map_l, error_map_r





# @cython.boundscheck(False) 
# @cython.wraparound(False) 
# cdef inline void compute_cost(np.uint8_t[:, :, ::1] bit_strings_l, 
#                   np.uint8_t[:, :, ::1] bit_strings_r, 
#                   np.uint16_t[:, :, ::1] error_map,
#                   int u, 
#                   int l1_cropped, 
#                   int l2_cropped,
#                   int max_disparity,
#                   int block_size_squared) nogil:

#     cdef int cost
#     cdef int v1, v2, d, i

#     for v1 in range(l1_cropped):
#         for d in range(max_disparity + 1):
#             cost = 0
#             v2 = v1 + d
#             if v2 < l2_cropped:
#                 for i in range(block_size_squared):
#                     if bit_strings_r[u, v1, i] != bit_strings_l[u, v2, i]:
#                         cost += 1
            
#                 error_map[u, v1, d] = cost
    


# @cython.boundscheck(False)
# @cython.wraparound(False) 
# def census_matching(np.uint8_t[:, :, ::1] bit_strings_l, 
#                     np.uint8_t[:, :, ::1] bit_strings_r,
#                     int h1_cropped,
#                     int l1_cropped,
#                     int l2_cropped,
#                     int block_size = 3, 
#                     int max_disparity = 64):

#     cdef int u
#     cdef int v1
#     cdef int d
#     cdef int v2
#     cdef int i
#     cdef int block_size_squared = block_size * block_size
#     cdef np.uint16_t cost

#     cdef np.uint16_t[:, :, ::1] error_map = np.full(
#         (h1_cropped, l1_cropped, max_disparity + 1),
#         np.iinfo(np.uint16).max,
#         dtype=np.uint16,
#     )

#     for u in prange(h1_cropped, nogil=True, schedule='static'):
#         compute_cost(bit_strings_l, bit_strings_r, error_map, u, l1_cropped, l2_cropped, max_disparity, block_size_squared)
    
#     return error_map