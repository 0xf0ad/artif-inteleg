#ifndef VECTOR_H_
#define VECTOR_H_

// THIS FILE IS OBSOLETE
// I MIGHT JUST REMOVE IT

#include <stdlib.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))

struct Vector{
		double* entries;
		size_t dimentions;
};

void init_vec(struct Vector* vec, size_t dim){
		vec->dimentions = dim;
		vec->entries = (double*) malloc(sizeof(double) * dim);
}

void free_vec(struct Vector* vec){
		vec->dimentions = 0;
		free(vec->entries);
}

// prform the dot product between two vectors
double dot_vec(struct Vector* vec1, struct Vector* vec2){
		// take the minimum number of elements to avoid reading unallocated memory
		size_t iters = min(vec1->dimentions, vec2->dimentions);
		// perform the dot operation
		double sum = 0;
		for(size_t i = 0; i != iters; i++)
				sum += vec1->entries[i] * vec2->entries[i];
		return sum;
}

#endif
