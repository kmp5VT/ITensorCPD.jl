/* Compile in MATLAB with: mex sparseStackl.c */
// Copyright (c) 2025, Chris Camano, Ethan Epperly

#include <math.h>
#include <stdlib.h>

static int ilog_base(long a, int d) {
    /* returns floor(log_d(a)) for a>=1, d>=2 */
    int out = 0;
    while (a > 0) { a /= d; out += 1; }
    return out - 1;
}

/* Unbiased uniform integer in [0, n) using rejection sampling */
static int uniform_int(int n) {
    /* n must be >= 1 */
    unsigned long limit = (unsigned long)RAND_MAX - ((unsigned long)RAND_MAX % (unsigned long)n);
    int r;
    do {
        r = rand();
    } while ((unsigned long)r > limit);
    return r % n;
}
void sparsestack(int d, int m, int zeta, double* vals, int* rows, int* colstarts  ){


    if (zeta > d) zeta = d;

    int nnz = m * zeta;
    long i;
    /* block sizes: d = q*zeta + r with 0 <= r < zeta */
    int q = d / zeta;
    int r = d % zeta;

    /* column pointers */
   *(colstarts + 0) = 0;
    for (int i = 1; i < m+1; ++i){
	    *(colstarts + i) = i*zeta;
	}

    /* sign magnitude and bit budget for rand() */
    const double a = 1.0 / sqrt((double)zeta);
    const int    bits_per_rand = ilog_base((long)RAND_MAX + 1L, 2);

    unsigned int sign_buf = 0U;
    int          bits_left = 0;

    int p = 0;
    for (int col = 0; col < m; ++col) {
        for (int j = 0; j < zeta; ++j, ++p) {

            /* compute start and size for block j */
            int size_j, start_j;
            if (j < r) {
                size_j = q + 1;
                start_j = j * (q + 1);
            } else {
                size_j = q;
                start_j = r * (q + 1) + (j - r) * q;
            }

            /* unbiased offset in [0, size_j) */
            int off = uniform_int((int)size_j);
            *(rows + p) = start_j + (int)off;

            /* reuse bits of rand() for the sign */
            if (bits_left == 0) {
                sign_buf = (unsigned int)rand();
                bits_left = bits_per_rand;
            }
            int s = (sign_buf & 1U) ? 1 : -1;
            sign_buf >>= 1;
            --bits_left;

            *(vals + p) = s * a;
            
        }
    }
}