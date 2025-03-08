#define PACK_SIZE 8
// #define BATCH_SIZE 1
#define SIMD_SIZE 32
#define GROUP_SIZE 64
#define PACKS_PER_THREAD 2
#define RESULTS_PER_SIMDGROUP 4
#define VALUES_PER_THREAD PACK_SIZE * PACKS_PER_THREAD
#define VALUES_PER_THREAD_QUARTER VALUES_PER_THREAD / 4
// #define SCALE_STEP_PER_THREAD GROUP_SIZE / VALUES_PER_THREAD
// #define BLOCK_SIZE SIMD_SIZE * VALUES_PER_THREAD
// constexpr int VALUES_PER_THREAD = PACK_SIZE * PACKS_PER_THREAD;
constexpr int BLOCK_SIZE = SIMD_SIZE * VALUES_PER_THREAD;
constexpr int SCALE_STEP_PER_THREAD = GROUP_SIZE / VALUES_PER_THREAD;

int row_offset = thread_position_in_grid.y * RESULTS_PER_SIMDGROUP;
W += thread_index_in_simdgroup * PACKS_PER_THREAD + row_offset * W_shape[1];
X += thread_index_in_simdgroup * VALUES_PER_THREAD;
result += thread_position_in_grid.y * RESULTS_PER_SIMDGROUP;


scales += thread_index_in_simdgroup / SCALE_STEP_PER_THREAD + row_offset * scales_shape[1];
biases += thread_index_in_simdgroup / SCALE_STEP_PER_THREAD + row_offset * scales_shape[1];

#if BATCH_SIZE > 1
    thread float _y[BATCH_SIZE * RESULTS_PER_SIMDGROUP] = {0.0};
    thread vec<float, BATCH_SIZE>* y = (thread vec<float, BATCH_SIZE>*)&_y;
#else
    thread float y[BATCH_SIZE * RESULTS_PER_SIMDGROUP] = {0.0};
#endif


int k = 0;
for(;k < W_shape[1];k+=(SIMD_SIZE * PACKS_PER_THREAD)) {
    // thread vec<half, VALUES_PER_THREAD> _x = *(device vec<half, VALUES_PER_THREAD>*)(X); // cannot play with vector pointers
    // thread float sum = 0;
    // #pragma clang unroll(full)
    // for(int i = 0;i < VALUES_PER_THREAD;i++) {
    //     sum += x[i];
    // }

    #if BATCH_SIZE > 1
        thread half4 x[BATCH_SIZE][PACKS_PER_THREAD * 2];
        #pragma unroll(BATCH_SIZE)
        for(int b = 0; b < BATCH_SIZE; b++) {
            #if (VALUES_PER_THREAD == 8)
                *((thread float4*)&x[b][0]) = *(device float4*)(X + b * X_shape[1]); // copy 8 values
            #elif (VALUES_PER_THREAD == 16)
                *((thread ulong4*)&x[b][0]) = *(device ulong4*)(X + b * X_shape[1]); // copy 16 values
            #endif
        }
    #else
        thread float xsum = 0.0;
        // #if (VALUES_PER_THREAD == 8)
        // // thread vec<half, VALUES_PER_THREAD> _x = *(device vec<half, VALUES_PER_THREAD>*)(X); // cannot play with vector pointers
        //     thread float4 _x = *(device float4*)(X); // copy 8 values
        // #elif (VALUES_PER_THREAD == 16)
        //     thread ulong4 _x = *(device ulong4*)(X); // copy 16 values
        // #endif
        thread float x[VALUES_PER_THREAD];
        // #pragma unroll(VALUES_PER_THREAD_QUARTER)
        for (int i = 0;i < VALUES_PER_THREAD;i+=4) {
            xsum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x[i] = X[i];
            x[i + 1] = X[i + 1] / 16.0f;
            x[i + 2] = X[i + 2] / 256.0f;
            x[i + 3] = X[i + 3] / 4096.0f;
        }
        // thread float* x = (thread float*)&_x;
    #endif
    
    for(int row = 0; row < RESULTS_PER_SIMDGROUP;row++) {
        const device half* sl = scales + row * scales_shape[1];
        const device half* bl = biases + row * scales_shape[1];
        // half s = sl[0];
        // half b = bl[0];
        #if BATCH_SIZE > 1
            thread vec<ushort, PACKS_PER_THREAD * 2> ws = *(device vec<ushort, PACKS_PER_THREAD * 2>*)(W + row * W_shape[1]);
        #else
            const device ushort* ws = (const device ushort*)(W + row * W_shape[1]);
            thread float accum = 0.0;
        #endif

        #pragma unroll(PACKS_PER_THREAD * 2)
        for(int u = 0; u < PACKS_PER_THREAD * 2;u++) {
            #if BATCH_SIZE > 1
                thread half4 w = half4(
                    // ((ws[u] & ushort(0x000f)) / 1.0h) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]],
                    // ((ws[u] & ushort(0x00f0)) / 16.0h) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]],
                    // ((ws[u] & ushort(0x0f00)) / 256.0h) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]],
                    // ((ws[u] & ushort(0xf000)) / 4096.0h) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]]
                    ((ws[u] & ushort(0x000f))) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]],
                    ((ws[u] & ushort(0x00f0))) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]],
                    ((ws[u] & ushort(0x0f00))) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]],
                    ((ws[u] & ushort(0xf000))) * scales[row * scales_shape[1]] + biases[row * scales_shape[1]]
            );
            #endif

            #if BATCH_SIZE > 1
                // thread matrix<float, BATCH_SIZE, 4> mt;
                thread matrix<float, BATCH_SIZE, 4> mt;
                #pragma unroll(BATCH_SIZE)
                for(int b = 0; b < BATCH_SIZE; b++) {
                    // m[b] sets the b-th column
                    // mt[b] = float4(x[b][u]);
                    // mt[b] = (x[b][u]);
                    // mt[b] = *((device half4*)(X + u * 4 + b * X_shape[1])) / half4(1.0h, 16.0h, 256.0h, 4096.h);
                    mt[b] = float4(*((device half4*)(X + u * 4 + b * X_shape[1])) / half4(1.0h, 16.0h, 256.0h, 4096.h));
                };
            #endif


            #if BATCH_SIZE > 1
                thread vec<float, BATCH_SIZE> matrixproduct = float4(w) * (mt); // vector * matrix because of format
                y[row] += matrixproduct;
                // thread vec<half, BATCH_SIZE> matrixproduct = w * mt; // vector * matrix because of format
                // y[row] += vec<float, BATCH_SIZE>(matrixproduct);
            #else
                // y[row] += dot(w, x[0][u]);
                // y[row] += biases[row * scales_shape[1]] * (float(x[u][0]) + float(x[u][1]) + float(x[u][2]) + float(x[u][3]));
                // accum += dot(float4(w), x[u]);
                accum += (
                    (ws[u] & 0x000f) * x[4 * u] +
                    (ws[u] & 0x00f0) * x[4 * u + 1] +
                    (ws[u] & 0x0f00) * x[4 * u + 2] +
                    (ws[u] & 0xf000) * x[4 * u + 3]
                );
            #endif
        }
        #if BATCH_SIZE == 1
            y[row] += accum * sl[0] + bl[0] * xsum;
        #endif
    }
    scales += BLOCK_SIZE / GROUP_SIZE;
    biases += BLOCK_SIZE / GROUP_SIZE;
    X += BLOCK_SIZE;
    W += PACKS_PER_THREAD * SIMD_SIZE;

}

// #pragma unroll(BATCH_SIZE)
// for (int b = 0; b < BATCH_SIZE; b++) {
// // }
// #pragma unroll(RESULTS_PER_SIMDGROUP)
for(int row = 0; row < RESULTS_PER_SIMDGROUP;row++) {
    y[row] = simd_sum(y[row]);
    // if (thread_index_in_simdgroup  == 0) {
    //     result[row] = y[row];
    // }
}

#if BATCH_SIZE > 1
thread matrix<float, RESULTS_PER_SIMDGROUP, BATCH_SIZE>* out_m = (thread matrix<float, RESULTS_PER_SIMDGROUP, BATCH_SIZE>*) y;
thread matrix<float, BATCH_SIZE, RESULTS_PER_SIMDGROUP> out_mt = transpose(*out_m);
#else
thread float4* out_mt = ((thread float4*)&y[0]);
#endif

// if (thread_index_in_simdgroup == RESULTS_PER_SIMDGROUP) {
if (thread_index_in_simdgroup == 0) {
    #pragma unroll(BATCH_SIZE)
    for (int b = 0; b < BATCH_SIZE; b++) {
        // *((device float4*)(result[b * W_shape[0] + thread_index_in_simdgroup])) = y[thread_index_in_simdgroup][b] + 1;
        thread half4 someval = half4(*((thread float4*)(&out_mt[b])));
        *((device half4*)(&result[b * W_shape[0] + thread_index_in_simdgroup])) = someval;
        // result[b * W_shape[0] + thread_index_in_simdgroup] = half(someval[0]) + thread_index_in_simdgroup;
    }
}