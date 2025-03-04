#define PACK_SIZE 8
#define SIMD_SIZE 32
#define GROUP_SIZE 64
#define PACKS_PER_THREAD 1
#define RESULTS_PER_SIMDGROUP 8
#define VALUES_PER_THREAD PACK_SIZE * PACKS_PER_THREAD
// #define SCALE_STEP_PER_THREAD GROUP_SIZE / VALUES_PER_THREAD
// #define BLOCK_SIZE SIMD_SIZE * VALUES_PER_THREAD
// constexpr int VALUES_PER_THREAD = PACK_SIZE * PACKS_PER_THREAD;
constexpr int BLOCK_SIZE = SIMD_SIZE * VALUES_PER_THREAD;
constexpr int SCALE_STEP_PER_THREAD = GROUP_SIZE / VALUES_PER_THREAD;

int row_offset = thread_position_in_grid.z * RESULTS_PER_SIMDGROUP;
W += thread_index_in_simdgroup * PACKS_PER_THREAD + row_offset * W_shape[1];
X += thread_index_in_simdgroup * VALUES_PER_THREAD;

scales += thread_index_in_simdgroup / SCALE_STEP_PER_THREAD + row_offset * scales_shape[1];
biases += thread_index_in_simdgroup / SCALE_STEP_PER_THREAD + row_offset * scales_shape[1];

thread float y[RESULTS_PER_SIMDGROUP] = {0};

int k = 0;
for(;k < W_shape[1];k+=(SIMD_SIZE * PACKS_PER_THREAD)) {
    #if (VALUES_PER_THREAD == 8)
    // thread vec<half, VALUES_PER_THREAD> _x = *(device vec<half, VALUES_PER_THREAD>*)(X); // cannot play with vector pointers
    thread float4 _x = *(device float4*)(X); // copy 8 values
    thread half4* x = (thread half4*)&_x;
    #endif
    // thread float sum = 0;
    // #pragma clang unroll(full)
    // for(int i = 0;i < VALUES_PER_THREAD;i++) {
    //     sum += x[i];
    // }
    for(int row = 0; row < RESULTS_PER_SIMDGROUP;row++) {
        thread vec<ushort, PACKS_PER_THREAD * 2> ws = *(device vec<ushort, PACKS_PER_THREAD * 2>*)(W + row * W_shape[1]);
        for(int u = 0; u < PACKS_PER_THREAD * 2;u++) {
            thread half4 w = half4(
                ((ws[u] & ushort(0x000f)) / 1.0h) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]],
                ((ws[u] & ushort(0x00f0)) / 16.0h) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]],
                ((ws[u] & ushort(0x0f00)) / 256.0h) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]],
                ((ws[u] & ushort(0xf000)) / 4096.0h) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]]
            );
            y[row] += dot(float4(x[u]), float4(w));
        }
    }
    scales += BLOCK_SIZE / GROUP_SIZE;
    biases += BLOCK_SIZE / GROUP_SIZE;
    X += BLOCK_SIZE;
    W += PACKS_PER_THREAD * SIMD_SIZE;
}

result += thread_position_in_grid.z * RESULTS_PER_SIMDGROUP;
for(int row = 0; row < RESULTS_PER_SIMDGROUP;row++) {
    y[row] = simd_sum(y[row]);
}
if (thread_index_in_simdgroup < RESULTS_PER_SIMDGROUP) {
    result[thread_index_in_simdgroup] = y[thread_index_in_simdgroup];
}