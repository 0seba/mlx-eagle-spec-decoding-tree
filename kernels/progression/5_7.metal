#define DEBUG 0
#define SIMDS_PER_TG 4
#define GROUP_SIZE 64
#define INNER_LOOP_EXPANSION 4
constexpr int SB_SIZE = (INNER_LOOP_EXPANSION * 32 >= GROUP_SIZE) ? (INNER_LOOP_EXPANSION * 32) / GROUP_SIZE : 1;
constexpr int INNER_LOOP_ITERATIONS = INNER_LOOP_EXPANSION / SB_SIZE;
// constexpr int group_size_s = GROUP_SIZE / int(8 * INNER_LOOP_EXPANSION);
constexpr int group_size_s = GROUP_SIZE / int(8);


// not to determine the W offset, 32 contigous elements of W contain 4 8x8 matrices, 32 elements advance by 32, thus we need K elements
// to advance by an Kx8 block-column
int K = X_shape[1]; 
W += K * thread_position_in_grid.z;

device uint* Wq = (device uint*)(W + thread_index_in_simdgroup);
constexpr int wq_step_size = 32;

int row = 8 * (thread_position_in_grid.z);
thread int leader = ((thread_index_in_simdgroup & 1) | (((thread_index_in_simdgroup >> 3) & 1) << 1));
scales += (row + leader * 2) * scales_shape[1];
biases += (row + leader * 2) * biases_shape[1];
const device half* scales2 = scales + scales_shape[1];
const device half* biases2 = biases + biases_shape[1];

#if DEBUG
debugW += 8 * thread_position_in_grid.z;
#endif


constexpr int lookup[32] = {
    0, 0, 1, 1, 2, 2, 3, 3,   // for x = 0..7
    0, 0, 1, 1, 2, 2, 3, 3,   // for x = 8..15
    4, 4, 5, 5, 6, 6, 7, 7,   // for x = 16..23
    4, 4, 5, 5, 6, 6, 7, 7    // for x = 24..31
};
X+=leader * 2 + lookup[thread_index_in_simdgroup] * X_shape[1];
// int offset = (((thread_index_in_simdgroup >> 4) & 1) << 2) | (((thread_index_in_simdgroup >> 2) & 1) << 1) | ((thread_index_in_simdgroup >> 1) & 1);
// X+=leader * 2 + offset * X_shape[1];

// simdgroup_matrix<half, 8, 8> sgMatrixW;
// simdgroup_matrix<half, 8, 8> sgMatrixX;
simdgroup_matrix<float, 8, 8> sgMatrixR = make_filled_simdgroup_matrix<float, 8, 8>(0.0);
// ------------------------------------------------------------------------------
// These are remnants from the kernel without swizzling to iterate over K, so I think should work here too
int k = 0;
int k_step_size = 4 * INNER_LOOP_EXPANSION;
// int iter_limit = W_shape[1] - k_step_size; // assume that K is divisible by 32 -- this was used when doing manual async_copy and wait
int iter_limit = W_shape[1];
// ------------------------------------------------------------------------------
for(; k < iter_limit; k+=k_step_size) {
    // half s0 = scales[k / group_size_s];
    // half s1 = scales2[k / group_size_s];
    // half b0 = biases[k / group_size_s];
    // half b1 = biases2[k / group_size_s];

    // we can know beforehand how many scales and biases will be needed for the inner loop
    // current swizzling and the fact that i have to unpack in reverse byte order means
    // that we can only load 1 contigous int at a time per thread (short isn't allowed because of inverse order)
    
    vec<half, SB_SIZE> s0s, s1s, b0s, b1s;
    s0s = *((device vec<half, SB_SIZE>*)(&scales[k / group_size_s]));
    s1s = *((device vec<half, SB_SIZE>*)(&scales2[k / group_size_s]));
    b0s = *((device vec<half, SB_SIZE>*)(&biases[k / group_size_s]));
    b1s = *((device vec<half, SB_SIZE>*)(&biases2[k / group_size_s]));

    #pragma unroll(SB_SIZE)
    for(int sb_index = 0; sb_index < SB_SIZE; sb_index++) {
        half s0 = s0s[sb_index];
        half s1 = s1s[sb_index];
        half b0 = b0s[sb_index];
        half b1 = b1s[sb_index];

        #pragma unroll(INNER_LOOP_ITERATIONS)
        for(int i = 0; i < INNER_LOOP_ITERATIONS; i++) {
            thread short2 quantized_weights = *((device short2*)Wq);
            
            thread half2 x[4];
            for (int j = 0; j < 4; j++) {
                x[j] = *((device half2*)&X[j * 8]);
            }

            #pragma unroll(4)
            for (ushort u = 0; u < 4; u++) {
                // thread half2 x = *((device half2*)&X[u * 8]);
                thread ushort packed_bits = quantized_weights[1 - u / 2]; // have to be reverted for some reason
                simdgroup_matrix<half, 8, 8> sgMatrixW;
                simdgroup_matrix<half, 8, 8> sgMatrixX;
                
                thread ushort wq0_0;
                thread ushort wq0_1;
                switch (u % 2) {
                    case 0:
                        wq0_0 = packed_bits >> 12;
                        wq0_1 = (packed_bits & ushort(0x0F00)) >> 8;
                        break;
                    case 1:
                        wq0_0 = (packed_bits & ushort(0x00F0)) >> 4;
                        wq0_1 = (packed_bits & ushort(0x000F));
                        break;
                }
                thread half w0_0 = static_cast<half>(wq0_0);
                thread half w0_1 = static_cast<half>(wq0_1);
                // sgMatrixW.thread_elements()[0] = w0_0;
                // sgMatrixW.thread_elements()[1] = w0_1;
                w0_0 = w0_0 * s0 + b0;
                w0_1 = w0_1 * s1 + b1;
                sgMatrixW.thread_elements()[0] = w0_0;
                sgMatrixW.thread_elements()[1] = w0_1;
                #if DEBUG
                simdgroup_store(sgMatrixW, debugW, W_shape[0]);
                debugW += 8 * W_shape[0];
                #endif
                // w0_1 = w0_1 * s1 / 16.0h + b1;
                *((thread half2*)&sgMatrixX.thread_elements()) = x[u];
                simdgroup_multiply_accumulate(sgMatrixR, sgMatrixX, sgMatrixW, sgMatrixR);
                // thread ushort packed_bits = quantized_weights[1 - u]; // have to be reverted for some reason
                // simdgroup_matrix<half, 8, 8> sgMatrixW[2];
                // simdgroup_matrix<half, 8, 8> sgMatrixX[2];
                
                // *((thread half2*)&sgMatrixX[0].thread_elements()) = *((device half2*)&X[u * 2 * 8]);
                // thread ushort wq0_0 = (packed_bits & 0xF000) >> 12;
                // thread ushort wq0_1 = (packed_bits & 0x0F00) >> 8;
                // thread half w0_0 = static_cast<half>(wq0_0);
                // thread half w0_1 = static_cast<half>(wq0_1);
                // w0_0 = w0_0 * s0 + b0;
                // w0_1 = w0_1 * s1 + b1;
                // // w0_1 = w0_1 * s1 / 16.0h + b1;
                // sgMatrixW[0].thread_elements()[0] = w0_0;
                // sgMatrixW[0].thread_elements()[1] = w0_1;
                // simdgroup_multiply_accumulate(sgMatrixR, sgMatrixX[0], sgMatrixW[0], sgMatrixR);
                
                // // simdgroup_store(sgMatrixW[0], &debugW[8 * (k + u * 2) * W_shape[0]], W_shape[0]);
                
                // *((thread half2*)&sgMatrixX[1].thread_elements()) = *((device half2*)&X[u * 2 * 8 + 8]);
                // thread ushort wq1_0 = (packed_bits & 0x00F0) >> 4;
                // thread ushort wq1_1 = (packed_bits & 0x000F);
                // thread half w1_0 = static_cast<half>(wq1_0);
                // thread half w1_1 = static_cast<half>(wq1_1);
                // w1_0 = w1_0 * s0 + b0;
                // w1_1 = w1_1 * s1 + b1;
                // // w1_0 = w1_0 * s0 / 256.0h + b0;
                // // w1_1 = w1_1 * s1 / 4096.0h + b1;
                // sgMatrixW[1].thread_elements()[0] = w1_0;
                // sgMatrixW[1].thread_elements()[1] = w1_1;
                // simdgroup_multiply_accumulate(sgMatrixR, sgMatrixX[1], sgMatrixW[1], sgMatrixR);

                // // simdgroup_store(sgMatrixW[1], &debugW[8 * (k + u * 2 + 1) * W_shape[0]], W_shape[0]);
            }
            X+=32 ;
            Wq += wq_step_size;
        }
    }
    // X+=32 * INNER_LOOP_EXPANSION;
    // Wq += wq_step_size * INNER_LOOP_EXPANSION;
}

simdgroup_matrix<half, 8, 8> sgMatrixDowncast;
sgMatrixDowncast.thread_elements()[0] = static_cast<half>(sgMatrixR.thread_elements()[0]);
sgMatrixDowncast.thread_elements()[1] = static_cast<half>(sgMatrixR.thread_elements()[1]);

simdgroup_store(sgMatrixDowncast, &result[8 * thread_position_in_grid.z], W_shape[0]);