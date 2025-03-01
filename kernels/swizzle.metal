// I want to implement a metal kernel to perform swizzling of an array of quantized weights
// The original weights W have shape KxN, and on quantization the weights are transposed
// and stored at 4-bit precision packed inside of a uint32 element, constructing
// a Mx(K/8) matrix of quantized weights with uint32 data type
// each element contains 8 weights, if those weights are represented as a vector with 8 elements, then
//      - the bits 0-3 contain the element 1
//      - the bits 4-7 contain the element 0
//      - the bits 8-11 contain the element 3 
//      - the bits 12-15 contain the element 2
//      - the bits 16-19 contain the element 7
//      - the bits 20-23 contain the element 6
//      - the bits 24-27 contain the element 5
//      - the bits 28-31 contain the element 4

// I want to swizzle them in a special format.
// First, I want a single SIMD group of 32 threads to read 32 consecutive ints, one each
// this means that for each read i will load 256 weights, which will represent 4 8x8 sub-matrices of the original matrix.
// The simdmatrices in apple metal have a special order based on the mordor order which is illustrated here

// The layout of threads within a SIMD matrix.
//
//  0  0  1  1  8  8  9  9
//  2  2  3  3 10 10 11 11
//  4  4  5  5 12 12 13 13
//  6  6  7  7 14 14 15 15
// 16 16 17 17 24 24 25 25
// 18 18 19 19 26 26 27 27
// 20 20 21 21 28 28 29 29
// 22 22 23 23 30 30 31 31
//
// This is Morton order, a method for coalescing data accesses. It is used
// in a variety of contexts, from ray tracing acceleration structures, to
// nodal-point Laplacians, to sorting large lattices of atoms.
//
// Source: https://patents.google.com/patent/US11256518B2

// For example this means that the first thread in the simdgroup contains the first 2 element of the matrix
// the thread with index 2 of the simdgroup will have to contain 2 first 2 values of the second row of the matrix
// As you can see THIS IS NOT LINEAR and has a very particular order.
// Additionally, remember that I told you that the quantized weights are transpoed, so in my swizzling I have to detranspose them
// I told you that each thread loads a int32 that contains 8 elements, equivalently to loading 4 8x8 matrices in each load,
// then I want that in the 8 dequantized elements that each thread loads, it has the 2 elements corresponding to 
// that threads position for each of the 4 matrices

// THIS IS IMPORTANT: I'm going to give you the quantized weights which come in a linear sequential order transposed and
// is not useful to me because I would have to perform cross-thread communication such that every thread
// receives its corresponding value in the morton order, but that is very expensive.
// And I want you to swizzle the quantized weights and the kernel you have to implement
// needs to change that linear order and swizzle the weights such that I do not perform cross-thread communication

// based on this layout and the fact that we have to transpose the weights
// if each thread loads 8 weights (packed as uint) and we advance loaded row every 4 threads
// in total per simdgroup we'll load 256 weights: 64x4, we have to do 4 iterations
// on the first iteration:
//      - thread 0 sends weights to threads (0, 2, 4, 6, 16, 18, 20, 22)
//      - thread 4 sends weights to threads (0, 2, 4, 6, 16, 18, 20, 22)
//      - thread 8 sends weights to threads (1, 3, 5, 7, 17, 19, 21, 23)
//      - thread 12 sends weights to threads (1, 3, 5, 7, 17, 19, 21, 23)
//      - thread 16 sends weights to threads (8, 10, 12, 14, 24, 26, 28, 30)
//      - thread 20 sends weights to threads (8, 10, 12, 14, 24, 26, 28, 30)
//      - thread 24 sends weights to threads (9, 11, 13, 15, 25, 27, 29, 31)
//      - thread 28 sends weights to threads (9, 11, 13, 15, 25, 27, 29, 31)
// on the second iteration:
//      - thread 1 sends weights to threads (0, 2, 4, 6, 16, 18, 20, 22)
//      - thread 5 sends weights to threads (0, 2, 4, 6, 16, 18, 20, 22)
//      - thread 9 sends weights to threads (1, 3, 5, 7, 17, 19, 21, 23)
//      - thread 13 sends weights to threads (1, 3, 5, 7, 17, 19, 21, 23)
//      - thread 17 sends weights to threads (8, 10, 12, 14, 24, 26, 28, 30)
//      - thread 21 sends weights to threads (8, 10, 12, 14, 24, 26, 28, 30)
//      - thread 25 sends weights to threads (9, 11, 13, 15, 25, 27, 29, 31)
//      - thread 29 sends weights to threads (9, 11, 13, 15, 25, 27, 29, 31)
// ...
// rule is
//          (0, 2, 4, 6, 16, 18, 20, 22) are mapped to 0
//          (1, 3, 5, 7, 17, 19, 21, 23) are mapped to 1
//          (8, 10, 12, 14, 24, 26, 28, 30) are mapped to 2
//          (9, 11, 13, 15, 25, 27, 29, 31) are mapped to 3
// ChatGPT solution:
//  We can solve this by noticing that the output depends only on two bits of the input: bit 0 (the least‐significant bit) and bit 3
// group 0  all have bit 0 = 0 and bit 3 = 0
// group 1  all have bit 0 = 1 and bit 3 = 0
// group 2  all have bit 0 = 0 and bit 3 = 1
// group 3  all have bit 0 = 1 and bit 3 = 1


// a single thread will load an (8x4) int tile
// this contains 256 that we have to rearange

Q += thread_index_in_quadgroup + (thread_index_in_simdgroup / 4 + threadgroup_position_in_grid.z * 8) * Q_shape[1];
thread ushort leader = ((thread_index_in_simdgroup & 1) | (((thread_index_in_simdgroup >> 3) & 1) << 1));
thread ushort shuffle_indexing = (((thread_index_in_simdgroup >> 4) & 1) << 1) | ((thread_index_in_simdgroup >> 2) & 1);
thread bool useLowNibble = ((thread_index_in_simdgroup & 3) < 2);

simdgroup_matrix<half, 8, 8> sgMatrixW;
// debugQ += 8 * threadgroup_position_in_grid.z;
Q_swizzled += 8 * threadgroup_position_in_grid.z * Q_shape[1] + thread_index_in_simdgroup;

for(int k=0;k<Q_shape[1];k+=4) {
    uint wq_uint = Q[k];
    thread uchar4 wq = *((thread uchar4*)&wq_uint);
    
    thread uint swizzled_w = 0;

    thread ushort iter_leader = leader * ushort(8);
    #pragma unroll
    for (ushort u = 0; u < 4; u++) {
        thread uchar wq0 = simd_shuffle(wq, iter_leader)[shuffle_indexing];
        thread uchar wq1 = simd_shuffle(wq, iter_leader + 4)[shuffle_indexing];

        wq0 = useLowNibble ? (wq0 & 0xF) : (wq0 >> 4);
        wq1 = useLowNibble ? (wq1 & 0xF) : (wq1 >> 4);

        // extract the corresponding byte from wq
        sgMatrixW.thread_elements()[0] = static_cast<half>(wq0);
        sgMatrixW.thread_elements()[1] = static_cast<half>(wq1);

        // simdgroup_store(sgMatrixW, debugQ, Q_shape[0]);

        swizzled_w = swizzled_w << 4;
        swizzled_w |= wq0;
        swizzled_w = swizzled_w << 4;
        swizzled_w |= wq1;

        iter_leader++;
        // debugQ+=8*Q_shape[0];
    }
    *Q_swizzled = swizzled_w;
    Q_swizzled+=32;
}
