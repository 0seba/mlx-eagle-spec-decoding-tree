// kernel specialized for group size 64, I'm not sure if other group sizes will work
// #define BATCH_SIZE 8
#define PACK_SIZE 8
#define SIMD_SIZE 32
#define GROUP_SIZE 64
#define ROWS_PER_SIMD 8
#define X_LOAD_SIZE_PER_THREAD 8
constexpr int PACKS_PER_THREAD = GROUP_SIZE / PACK_SIZE;
constexpr int VALUES_PER_THREAD = GROUP_SIZE;
constexpr int SCALE_STEP_PER_THREAD = 4;
constexpr int NUM_BATCH_CHUNKS = BATCH_SIZE / 8;
// we'll split compute in chunks of K of size 256 (64 values and 4 thread for the same K row)
// and handle the residuals of size 128, meaning that this kernel only works for K multiple of 128, which will serve
// most LLM dimensions

// threadgroup half* cast_int4_sh = &(_cast_int4 + 16 * threadgroup_position_in_grid.z);

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
/* leaders = {
    0, 1, 0, 1, 0, 1, 0, 1, // 0..7
    2, 3, 2, 3, 2, 3, 2, 3, // 8..15
    0, 1, 0, 1, 0, 1, 0, 1, // 16..23
    2, 3, 2, 3, 2, 3, 2, 3  // 24..31
}
*/
thread int leader = ((thread_index_in_simdgroup & 1) | (((thread_index_in_simdgroup >> 3) & 1) << 1));
constexpr int lookup[32] = {
    0, 0, 1, 1, 2, 2, 3, 3,   // for x = 0..7
    0, 0, 1, 1, 2, 2, 3, 3,   // for x = 8..15
    4, 4, 5, 5, 6, 6, 7, 7,   // for x = 16..23
    4, 4, 5, 5, 6, 6, 7, 7    // for x = 24..31
};
int lu = lookup[thread_index_in_simdgroup];
// int lookup = (((thread_index_in_simdgroup >> 4) & 1) << 2) | (((thread_index_in_simdgroup >> 2) & 1) << 1) | ((thread_index_in_simdgroup >> 1) & 1);
ushort w_row = ROWS_PER_SIMD * thread_position_in_grid.z + lu;

W += leader * PACKS_PER_THREAD + w_row * W_shape[1];
X += leader * 2 * X_shape[1] + 4 * (lu % 2) + GROUP_SIZE * (lu / 2);

const device half* X1[NUM_BATCH_CHUNKS];
const device half* X2[NUM_BATCH_CHUNKS];

for(ushort i = 0; i < NUM_BATCH_CHUNKS; i++) {
    X1[i] = X + i * 8 * X_shape[1];
    X2[i] = X1[i] + X_shape[1];
}

const device half* s = scales + w_row * scales_shape[1] + leader;
const device half* b = biases + w_row * scales_shape[1] + leader;

simdgroup_matrix<float, 8, 8> sgMatrixR[NUM_BATCH_CHUNKS];
for(ushort i = 0; i < NUM_BATCH_CHUNKS; i++) {
    sgMatrixR[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0);
}

// ushort num_iters = (X_shape[1] / 256) - 1;
ushort num_iters = (X_shape[1] / 256);
ushort remainder = (X_shape[1] % 256) > 0;
// for(ushort k = 0; k <= num_iters; k++) {
for(ushort k = 0; k < num_iters; k++) {
    half s0 = *s;
    half b0 = *b;

    #pragma unroll(PACKS_PER_THREAD)
    for (ushort i = 0; i < PACKS_PER_THREAD; i++) {
        uint w = *W;
        half2 w1 = static_cast<half2>(as_type<ushort2>(w & 0x000f000f));
        w1 = w1 * s0 + b0;
        simdgroup_matrix<half, 8, 8> sgMatrixW1;
        sgMatrixW1.thread_elements()[0] = w1.x;
        sgMatrixW1.thread_elements()[1] = w1.y;

        #pragma unroll(NUM_BATCH_CHUNKS)
        for (ushort j = 0; j < NUM_BATCH_CHUNKS; j++) {
            simdgroup_matrix<half, 8, 8> sgMatrixX1;
            sgMatrixX1.thread_elements()[0] = X1[j][0];
            sgMatrixX1.thread_elements()[1] = X2[j][0];
            simdgroup_multiply_accumulate(sgMatrixR[j], sgMatrixW1, sgMatrixX1, sgMatrixR[j]);
        }

        half2 w2 = static_cast<half2>(as_type<ushort2>((w & 0x00f000f0) >> 4));
        w2 = w2 * s0 + b0;
        simdgroup_matrix<half, 8, 8> sgMatrixW2;
        sgMatrixW2.thread_elements()[0] = w2.x;
        sgMatrixW2.thread_elements()[1] = w2.y;

        #pragma unroll(NUM_BATCH_CHUNKS)
        for (ushort j = 0; j < NUM_BATCH_CHUNKS; j++) {
            simdgroup_matrix<half, 8, 8> sgMatrixX2;
            sgMatrixX2.thread_elements()[0] = X1[j][1];
            sgMatrixX2.thread_elements()[1] = X2[j][1];
            simdgroup_multiply_accumulate(sgMatrixR[j], sgMatrixW2, sgMatrixX2, sgMatrixR[j]);
        }

        half2 w3 = static_cast<half2>(as_type<ushort2>((w & 0x0f000f00) >> 8));
        w3 = w3 * s0 + b0;
        simdgroup_matrix<half, 8, 8> sgMatrixW3;
        sgMatrixW3.thread_elements()[0] = w3.x;
        sgMatrixW3.thread_elements()[1] = w3.y;

        #pragma unroll(NUM_BATCH_CHUNKS)
        for (ushort j = 0; j < NUM_BATCH_CHUNKS; j++) {
            simdgroup_matrix<half, 8, 8> sgMatrixX3;
            sgMatrixX3.thread_elements()[0] = X1[j][2];
            sgMatrixX3.thread_elements()[1] = X2[j][2];
            simdgroup_multiply_accumulate(sgMatrixR[j], sgMatrixW3, sgMatrixX3, sgMatrixR[j]);
        }

        half2 w4 = static_cast<half2>(as_type<ushort2>((w & 0xf000f000) >> 12));
        w4 = w4 * s0 + b0;
        simdgroup_matrix<half, 8, 8> sgMatrixW4;
        simdgroup_matrix<half, 8, 8> sgMatrixX4;
        sgMatrixW4.thread_elements()[0] = w4.x;
        sgMatrixW4.thread_elements()[1] = w4.y;

        #pragma unroll(NUM_BATCH_CHUNKS)
        for (ushort j = 0; j < NUM_BATCH_CHUNKS; j++) {
            simdgroup_matrix<half, 8, 8> sgMatrixX4;
            sgMatrixX4.thread_elements()[0] = X1[j][3];
            sgMatrixX4.thread_elements()[1] = X2[j][3];
            simdgroup_multiply_accumulate(sgMatrixR[j], sgMatrixW4, sgMatrixX4, sgMatrixR[j]);
        }

        #pragma unroll(NUM_BATCH_CHUNKS)
        for (ushort j = 0; j < NUM_BATCH_CHUNKS; j++) {
            X1[j] += 8;
            X2[j] += 8;
        }
        W += 1;
    }
    W += PACKS_PER_THREAD * 3;
    #pragma unroll(NUM_BATCH_CHUNKS)
    for(ushort j = 0; j < NUM_BATCH_CHUNKS; j++) {
        X1[j] += GROUP_SIZE * 3;
        X2[j] += GROUP_SIZE * 3;
    }
    // s += (k == num_iters) ? 0 : 4;
    // b += (k == num_iters) ? 0 : 4;
    s += 4;
    b += 4;
}

if(remainder) {
    s -= (leader + 1) / 2;
    b -= (leader + 1) / 2;
    W -= leader * (PACKS_PER_THREAD / 2);
    for(ushort j = 0; j < NUM_BATCH_CHUNKS; j++) {
        X1[j] -= (lu / 2) * (VALUES_PER_THREAD / 2);
        X2[j] -= (lu / 2) * (VALUES_PER_THREAD / 2);
    }
    half s0 = *s;
    half b0 = *b;
    
    for (ushort i = 0; i < PACKS_PER_THREAD / 2; i++) {
        uint w = *W;

        #pragma unroll(NUM_BATCH_CHUNKS)
        for(ushort j = 0; j < NUM_BATCH_CHUNKS; j++){
            thread half4 xs1 = *reinterpret_cast<const device half4*>(X1[j]);
            thread half4 xs2 = *reinterpret_cast<const device half4*>(X2[j]);

            half2 w1 = static_cast<half2>(as_type<ushort2>(w & 0x000f000f));
            w1 = w1 * s0 + b0;
            simdgroup_matrix<half, 8, 8> sgMatrixW1;
            simdgroup_matrix<half, 8, 8> sgMatrixX1;
            sgMatrixW1.thread_elements()[0] = w1.x;
            sgMatrixW1.thread_elements()[1] = w1.y;
            sgMatrixX1.thread_elements()[0] = xs1[0];
            sgMatrixX1.thread_elements()[1] = xs2[0];
            simdgroup_multiply_accumulate(sgMatrixR[j], sgMatrixW1, sgMatrixX1, sgMatrixR[j]);

            // simdgroup_store(sgMatrixW1, debugW + 4 * 8 * i * 8, 8, 0, true);
            // simdgroup_store(sgMatrixX1, debugX + 4 * 8 * i * 8, 8, 0, false);

            half2 w2 = static_cast<half2>(as_type<ushort2>((w & 0x00f000f0) >> 4));
            w2 = w2 * s0 + b0;
            simdgroup_matrix<half, 8, 8> sgMatrixW2;
            simdgroup_matrix<half, 8, 8> sgMatrixX2;
            sgMatrixW2.thread_elements()[0] = w2.x;
            sgMatrixW2.thread_elements()[1] = w2.y;
            sgMatrixX2.thread_elements()[0] = xs1[1];
            sgMatrixX2.thread_elements()[1] = xs2[1];
            simdgroup_multiply_accumulate(sgMatrixR[j], sgMatrixW2, sgMatrixX2, sgMatrixR[j]);

            // simdgroup_store(sgMatrixW2, debugW + (4 * 8 * i + 8) * 8, 8, 0, true);
            // simdgroup_store(sgMatrixX2, debugX + (4 * 8 * i + 8) * 8, 8, 0, false);

            half2 w3 = static_cast<half2>(as_type<ushort2>((w & 0x0f000f00) >> 8));
            w3 = w3 * s0 + b0;
            simdgroup_matrix<half, 8, 8> sgMatrixW3;
            simdgroup_matrix<half, 8, 8> sgMatrixX3;
            sgMatrixW3.thread_elements()[0] = w3.x;
            sgMatrixW3.thread_elements()[1] = w3.y;
            sgMatrixX3.thread_elements()[0] = xs1[2];
            sgMatrixX3.thread_elements()[1] = xs2[2];
            simdgroup_multiply_accumulate(sgMatrixR[j], sgMatrixW3, sgMatrixX3, sgMatrixR[j]);

            // simdgroup_store(sgMatrixW3, debugW + (4 * 8 * i + 16) * 8, 8, 0, true);
            // simdgroup_store(sgMatrixX3, debugX + (4 * 8 * i + 16) * 8, 8, 0, false);

            half2 w4 = static_cast<half2>(as_type<ushort2>((w & 0xf000f000) >> 12));
            w4 = w4 * s0 + b0;
            simdgroup_matrix<half, 8, 8> sgMatrixW4;
            simdgroup_matrix<half, 8, 8> sgMatrixX4;
            sgMatrixW4.thread_elements()[0] = w4.x;
            sgMatrixW4.thread_elements()[1] = w4.y;
            sgMatrixX4.thread_elements()[0] = xs1[3];
            sgMatrixX4.thread_elements()[1] = xs2[3];
            simdgroup_multiply_accumulate(sgMatrixR[j], sgMatrixW4, sgMatrixX4, sgMatrixR[j]);

            // simdgroup_store(sgMatrixW4, debugW + (4 * 8 * i + 24) * 8, 8, 0, true);
            // simdgroup_store(sgMatrixX4, debugX + (4 * 8 * i + 24) * 8, 8, 0, false);
            X1[j] += 8;
            X2[j] += 8;
        }

        W += 1;
    }
}

for(ushort i = 0; i < NUM_BATCH_CHUNKS; i++) {
    simdgroup_matrix<half, 8, 8> sgMatrixDowncast;
    sgMatrixDowncast.thread_elements()[0] = static_cast<half>(sgMatrixR[i].thread_elements()[0]);
    sgMatrixDowncast.thread_elements()[1] = static_cast<half>(sgMatrixR[i].thread_elements()[1]);
    simdgroup_store(sgMatrixDowncast, &result[8 * thread_position_in_grid.z + i * 8 * W_shape[0]], W_shape[0], 0, true);
}