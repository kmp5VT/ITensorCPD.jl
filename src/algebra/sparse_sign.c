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

void sparse_sign(int d, int m, int zeta, double* vals, int* rows, int* colstarts  )
{
    if (zeta > d) zeta = d;
    int idx_per_rand = log_base_d((long) RAND_MAX + 1, d);
    int bit_per_rand = log_base_d((long) RAND_MAX + 1, 2);

	double lowval = -1/sqrt((double) zeta);
    double increment = -2*lowval;

    long nnz=(long)m*zeta;
	// Set values
    int myrand = rand();
    long i;
	for (int i = 0; i+bit_per_rand < nnz; i += bit_per_rand) {
        for (int j = i; j < i+bit_per_rand; ++j) {
	        vals[j] = (myrand % 2) * increment + lowval;
            myrand = myrand >> 1;
        }
        myrand = rand();
	}
    for (int i = bit_per_rand*(nnz/bit_per_rand); i < nnz; ++i) {
        vals[i] = (myrand % 2) * increment + lowval;
        myrand = myrand >> 1;
    }

	// Set column starts
	for (int i = 1; i < m+1; ++i){
	    colstarts[i] = i*zeta;
	}

	// Set row indices
    myrand = rand();
    int ir = 0;
    for (int i = 0; i < m*zeta; i += zeta) {
	    int idx = 0;
	    while (idx < zeta) {
		    rows[i+idx] = myrand % d;
            ir++;
            if (ir == idx_per_rand) {
                ir = 0;
                myrand = rand();
            } else {
                myrand /= d;
            }
		    int j = 0;
		    for (; j < idx; ++j) {
		        if (rows[i+idx] == rows[i+j]) break;
		    }
		    idx += (int) (j == idx);
	    }
	}
}