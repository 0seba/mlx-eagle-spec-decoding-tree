uint M = Qswizzled_shape[0];
uint K = Qswizzled_shape[1] * 8;
simdgroup_matrix<half, 8, 8> sgMatrixW;

for(int i = 0; i<(Qswizzled_shape[0] * Qswizzled_shape[1]);i+=32) {
    thread uint wqs_uint = Qswizzled[thread_index_in_threadgroup + i];
    thread uchar4 wqs = *((thread uchar4*)&wqs_uint);
    for(int j=0;j<4;j++) { // idk why tf this has to be reversed
        thread uchar wq = wqs[3 - j];
        thread half w0 = static_cast<half>(wq >> 4);
        thread half w1 = static_cast<half>(wq & 0x0F);
        *((thread half2*)&sgMatrixW.thread_elements()) = half2(w0, w1);
        // for every j we advance 8 rows
        // for every i we advance 32 rows
        // circle around the rows to columns when advancing more than K rows
        thread int index = 8 * j + i;
        thread int row = index % K;
        thread int column = 8 * (index / K);
        simdgroup_store(sgMatrixW, Q + row * M + column, M);
    }
}