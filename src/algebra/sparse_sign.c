#include <math.h> 
#include <stdlib.h>
#include <stdio.h>

#define RADEMACHER() (2 * (rand() % 2) -1)

int log_base_d(long a, int d) {
    int output = 0;
    while (a > 0) {
        a /= d;
        output += 1;
    }
    return output-1;
}
/* Unbiased uniform integer in [0, n) using rejection sampling */
static int uniform_int(int n) {
    /* n must be >= 1 */
    long limit = (long)RAND_MAX - ((long)RAND_MAX % (long)n);
    int r;
    do {
        r = rand();
    } while ((unsigned long)r > limit);
    return r%n;
}
void sparse_sign(int d, int m, int zeta, double* vals, int* rows, int* colstarts  )
{
    if (zeta > d) zeta = d;
    int bit_per_rand = log_base_d((long) RAND_MAX + 1, 2);

	double lowval = 1/sqrt((double) zeta);

    long nnz=(long)m*zeta;

	// Set values
    unsigned int my_rand = rand();
    long i;
	for (int i = 0; i+bit_per_rand < nnz; i += bit_per_rand) {
        for (int j = i; j < i+bit_per_rand; ++j) {
            int sign = (my_rand & 1U)? 1: -1;
            *(vals + j) = sign*lowval;
            my_rand = my_rand >> 1;
        }
        my_rand = (unsigned int) rand();
	}
    for (int i = bit_per_rand*(nnz/bit_per_rand); i < nnz; ++i) {
        int sign = (my_rand & 1U)? 1: -1;
        *(vals + i) = sign*lowval;
        my_rand = my_rand >> 1;
    }

	// Set column starts
    *(colstarts + 0) = 0;
	for (int i = 1; i < m+1; ++i){
	    *(colstarts + i) = i*zeta;
	}

	// Set row indices
    for (int i = 0; i < m*zeta; i += zeta) {
	    int idx = 0;
	    while (idx < zeta) {
            int off = uniform_int(d);
		    *(rows + i+idx) = (int)off ;
		    int j = 0;
		    for (; j < idx; ++j) {
		        if (*(rows + i+idx) == *(rows + i+j)) break;
		    }
		    idx += (int) (j == idx);
	    }
	}
}