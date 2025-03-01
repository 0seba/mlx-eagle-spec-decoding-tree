#define MULT 1
#define SIMDS_PER_TG 4
#define GROUP_SIZE 64
threadgroup uint shmemwq[2][SIMDS_PER_TG][32 * MULT]; // double buffering up to 8 simdgroups/tgroup 1KB per buffer (should increase to fetch more weights)
threadgroup uint* shmemwq_ptr = (threadgroup uint*)shmemwq;

shmemwq_ptr += simdgroup_index_in_threadgroup * 32 * MULT;
// not to determine the W offset, 32 contigous elements of W contain 4 8x8 matrices, 32 elements advance by 32, thus we need K elements
// to advance by an Kx8 block-column
int K = X_shape[1]; 
W += K * thread_position_in_grid.z;

device uint* Wq = (device uint*)(W + thread_index_in_simdgroup);

// prefetch weights
metal::simdgroup_event active_weights_e;
metal::simdgroup_event inactive_weights_e;
// threadgroup uint* shmemwq_ptr_write = (threadgroup uint*)shmemwq;
threadgroup uint* shmemwq_ptr_write = shmemwq_ptr;
threadgroup uint* shmemwq_ptr_read = shmemwq_ptr_write + 32 * SIMDS_PER_TG * MULT;
// active_weights_e.async_copy(shmemwq_ptr_write, Wq, ulong(32 * MULT));
int wq_step_size = 32 * MULT;

// metal::simdgroup_event active_e; // double buffer
// metal::simdgroup_event inactive_e;
// ushort X_dst_elements_per_row = 32;
// ushort2 X_dst_tile_dimensions = ushort2(32, 8); // seems these are column-row order
// uint X_src_elements_per_row = X_shape[1];
// ushort2 X_src_tile_dimensions = ushort2(32, 8); // seems these are column-row order
// threadgroup half* shmemx_ptr_write = (threadgroup half*)shmemx;
// threadgroup half* shmemx_ptr_read = shmemx_ptr_write + 256;
// if (simdgroup_index_in_threadgroup == 0) {
//     active_e.async_copy(shmemx_ptr_write, X_dst_elements_per_row, X_dst_tile_dimensions, X, X_src_elements_per_row, X_src_tile_dimensions);
// }
// X+=32;
// thread uchar buffer_index = 1;
// metal::simdgroup_event inactive_event = activations_e_1;


int row = 8 * (thread_position_in_grid.z);
thread int leader = ((thread_index_in_simdgroup & 1) | (((thread_index_in_simdgroup >> 3) & 1) << 1));
scales += (row + leader * 2) * scales_shape[1];
biases += (row + leader * 2) * biases_shape[1];
const device half* scales2 = scales + scales_shape[1];
const device half* biases2 = biases + biases_shape[1];
constexpr int group_size_s = GROUP_SIZE / int(8 * MULT);

// debugW += 8 * thread_position_in_grid.z;


// constexpr int lookup[32] = {
//     0, 0, 1, 1, 2, 2, 3, 3,   // for x = 0..7
//     0, 0, 1, 1, 2, 2, 3, 3,   // for x = 8..15
//     4, 4, 5, 5, 6, 6, 7, 7,   // for x = 16..23
//     4, 4, 5, 5, 6, 6, 7, 7    // for x = 24..31
// };
// X+=leader * 2 + lookup[thread_index_in_simdgroup] * X_shape[1];
int offset = (((thread_index_in_simdgroup >> 4) & 1) << 2) | (((thread_index_in_simdgroup >> 2) & 1) << 1) | ((thread_index_in_simdgroup >> 1) & 1);
X+=leader * 2 + offset * X_shape[1];

// half cast_int4_inline[16] = {
//     0.0, 1.0, 2.0, 3.0,
//     4.0, 5.0, 6.0, 7.0,
//     8.0, 9.0, 10.0, 11.0,
//     12.0, 13.0, 14.0, 15.0
// };

// simdgroup_matrix<half, 8, 8> sgMatrixW;
// simdgroup_matrix<half, 8, 8> sgMatrixX;
simdgroup_matrix<float, 8, 8> sgMatrixR = make_filled_simdgroup_matrix<float, 8, 8>(0.0);
// ------------------------------------------------------------------------------
// These are remnants from the kernel without swizzling to iterate over K, so I think should work here too
int k = 0;
int k_step_size = 4 * MULT;
int iter_limit = W_shape[1] - k_step_size; // assume that K is divisible by 32
// ------------------------------------------------------------------------------
for(; k < iter_limit; k+=k_step_size) {
    half s0 = scales[k / group_size_s];
    half s1 = scales2[k / group_size_s];
    half b0 = biases[k / group_size_s];
    half b1 = biases2[k / group_size_s];

    // initialize new transfer to shared
    {
        threadgroup uint* tmp = shmemwq_ptr_write;
        shmemwq_ptr_write = shmemwq_ptr_read;
        shmemwq_ptr_read = tmp;
        // inactive_weights_e.async_copy(shmemwq_ptr_write, Wq, ulong(32 * MULT));
        thread simdgroup_event* e_helper = &inactive_weights_e;
        inactive_weights_e = active_weights_e;
        active_weights_e = *e_helper;
    }

    // simdgroup_matrix<half, 8, 8> sgMatrixX[4];
    // for (ushort u = 0; u < 4; u++) {
    //     // is this async? tryng doing before weights await
    //     simdgroup_load(sgMatrixX[u], &X[u*8], X_shape[1]);
    // }
    // metal::simdgroup_event::wait(1, &inactive_weights_e);
    // simdgroup_barrier(mem_flags::mem_threadgroup);

    // thread ushort2 quantized_weights = as_type<ushort2>(*(shmemwq_ptr_read + thread_index_in_simdgroup));
    thread ushort2 quantized_weights = *((device ushort2*)Wq);
    Wq += wq_step_size;

    #pragma unroll(2)
    for (ushort u = 0; u < 2; u++) {
        thread ushort packed_bits = quantized_weights[1 - u];
        simdgroup_matrix<half, 8, 8> sgMatrixW[2];
        simdgroup_matrix<half, 8, 8> sgMatrixX[2];
        
        // simdgroup_load(sgMatrixX[0], &X[u * 2 * 8], X_shape[1]);
        *((thread half2*)&sgMatrixX[0].thread_elements()) = *((device half2*)&X[u * 2 * 8]);
        thread ushort wq0_0 = (packed_bits & 0xF000) >> 12;
        thread ushort wq0_1 = (packed_bits & 0x0F00) >> 8;
        thread half w0_0 = static_cast<half>(wq0_0);
        thread half w0_1 = static_cast<half>(wq0_1);
        w0_0 = w0_0 * s0 + b0;
        w0_1 = w0_1 * s1 + b1;
        // w0_1 = w0_1 * s1 / 16.0h + b1;
        sgMatrixW[0].thread_elements()[0] = w0_0;
        sgMatrixW[0].thread_elements()[1] = w0_1;
        simdgroup_multiply_accumulate(sgMatrixR, sgMatrixX[0], sgMatrixW[0], sgMatrixR);
        
        // simdgroup_store(sgMatrixW[0], &debugW[8 * (k + u * 2) * W_shape[0]], W_shape[0]);
        
        // simdgroup_load(sgMatrixX[1], &X[u * 2 * 8 + 8], X_shape[1]);
        *((thread half2*)&sgMatrixX[1].thread_elements()) = *((device half2*)&X[u * 2 * 8 + 8]);
        thread ushort wq1_0 = (packed_bits & 0x00F0) >> 4;
        thread ushort wq1_1 = (packed_bits & 0x000F);
        thread half w1_0 = static_cast<half>(wq1_0);
        thread half w1_1 = static_cast<half>(wq1_1);
        w1_0 = w1_0 * s0 + b0;
        w1_1 = w1_1 * s1 + b1;
        // w1_0 = w1_0 * s0 / 256.0h + b0;
        // w1_1 = w1_1 * s1 / 4096.0h + b1;
        sgMatrixW[1].thread_elements()[0] = w1_0;
        sgMatrixW[1].thread_elements()[1] = w1_1;
        simdgroup_multiply_accumulate(sgMatrixR, sgMatrixX[1], sgMatrixW[1], sgMatrixR);

        // simdgroup_store(sgMatrixW[1], &debugW[8 * (k + u * 2 + 1) * W_shape[0]], W_shape[0]);
    }
    X+=32;
}
{
    half s0 = scales[k / group_size_s];
    half s1 = scales2[k / group_size_s];
    half b0 = biases[k / group_size_s];
    half b1 = biases2[k / group_size_s];

    // initialize new transfer to shared
    // Wq += wq_step_size;
    {
        threadgroup uint* tmp = shmemwq_ptr_write;
        shmemwq_ptr_write = shmemwq_ptr_read;
        shmemwq_ptr_read = tmp;
        // inactive_weights_e.async_copy(shmemwq_ptr_write, Wq, ulong(32 * MULT));
        thread simdgroup_event* e_helper = &inactive_weights_e;
        inactive_weights_e = active_weights_e;
        active_weights_e = *e_helper;
    }

    // simdgroup_matrix<half, 8, 8> sgMatrixX[4];
    // for (ushort u = 0; u < 4; u++) {
    //     // is this async? tryng doing before weights await
    //     simdgroup_load(sgMatrixX[u], &X[u*8], X_shape[1]);
    // }
    // metal::simdgroup_event::wait(1, &inactive_weights_e);
    // simdgroup_barrier(mem_flags::mem_threadgroup);
    // thread ushort2 quantized_weights = as_type<ushort2>(*(shmemwq_ptr_read + thread_index_in_simdgroup));
    thread ushort2 quantized_weights = *((device ushort2*)Wq);

    for (ushort u = 0; u < 2; u++) {
        thread ushort packed_bits = quantized_weights[1 - u];
        simdgroup_matrix<half, 8, 8> sgMatrixW[2];
        simdgroup_matrix<half, 8, 8> sgMatrixX[2];
        
        // simdgroup_load(sgMatrixX[0], &X[u * 2 * 8], X_shape[1]);
        *((thread half2*)&sgMatrixX[0].thread_elements()) = *((device half2*)&X[u * 2 * 8]);
        thread ushort wq0_0 = (packed_bits & 0xF000) >> 12;
        thread ushort wq0_1 = (packed_bits & 0x0F00) >> 8;
        thread half w0_0 = static_cast<half>(wq0_0);
        thread half w0_1 = static_cast<half>(wq0_1);
        w0_0 = w0_0 * s0 + b0;
        w0_1 = w0_1 * s1 + b1;
        // w0_1 = w0_1 * s1 / 16.0h + b1;
        sgMatrixW[0].thread_elements()[0] = w0_0;
        sgMatrixW[0].thread_elements()[1] = w0_1;
        simdgroup_multiply_accumulate(sgMatrixR, sgMatrixX[0], sgMatrixW[0], sgMatrixR);

        // simdgroup_store(sgMatrixW[0], &debugW[8 * (k + u * 2) * W_shape[0]], W_shape[0]);

        // simdgroup_load(sgMatrixX[1], &X[u * 2 * 8 + 8], X_shape[1]);
        *((thread half2*)&sgMatrixX[1].thread_elements()) = *((device half2*)&X[u * 2 * 8 + 8]);
        thread ushort wq1_0 = (packed_bits & 0x00F0) >> 4;
        thread ushort wq1_1 = (packed_bits & 0x000F);
        thread half w1_0 = static_cast<half>(wq1_0);
        thread half w1_1 = static_cast<half>(wq1_1);
        w1_0 = w1_0 * s0 + b0;
        w1_1 = w1_1 * s1 + b1;
        // w1_0 = w1_0 * s0 / 256.0h + b0;
        // w1_1 = w1_1 * s1 / 4096.0h + b1;
        sgMatrixW[1].thread_elements()[0] = w1_0;
        sgMatrixW[1].thread_elements()[1] = w1_1;
        simdgroup_multiply_accumulate(sgMatrixR, sgMatrixX[1], sgMatrixW[1], sgMatrixR);
        
        // simdgroup_store(sgMatrixW[1], &debugW[8 * (k + u * 2 + 1) * W_shape[0]], W_shape[0]);
    }
    X+=32;
}

simdgroup_matrix<half, 8, 8> sgMatrixDowncast;
sgMatrixDowncast.thread_elements()[0] = static_cast<half>(sgMatrixR.thread_elements()[0]);
sgMatrixDowncast.thread_elements()[1] = static_cast<half>(sgMatrixR.thread_elements()[1]);

simdgroup_store(sgMatrixDowncast, &result[8 * thread_position_in_grid.z], W_shape[0]);


// constexpr int lookup[32] = {
//     0, 0, 1, 1, 2, 2, 3, 3,   // for x = 0..7
//     0, 0, 1, 1, 2, 2, 3, 3,   // for x = 8..15
//     4, 4, 5, 5, 6, 6, 7, 7,   // for x = 16..23
//     4, 4, 5, 5, 6, 6, 7, 7    // for x = 24..31
// };

// *((device half2*)&result[leader * 2 + lookup[thread_index_in_simdgroup] * W_shape[0]]) = static_cast<half2>(
//     *((thread float2*)&sgMatrixR.thread_elements())
// );