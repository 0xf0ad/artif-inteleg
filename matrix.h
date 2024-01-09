#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <time.h>

struct Matrix{
	size_t rows;
	size_t colums;
	double* entries;
};

// params: pointer to the matrix to be initialized and it's rows and colums
inline void init_mat(struct Matrix* mat, size_t rows, size_t colums){
	if(!rows || !colums)
		assert(0);
	mat->rows = rows;
	mat->colums = colums;
	mat->entries = (double*) malloc(sizeof(double) * rows * colums);
}

// fill the entire matrix with one single value
inline void fill_mat(struct Matrix* mat, double val){
	size_t iters = mat->rows * mat->colums;
	for(size_t i = 0; i != iters; i++)
		mat->entries[i] = val;
}

// get a single element from the matrix based on its row and colum
inline double get_element(const struct Matrix* mat, size_t row, size_t colum){
	//if(mat->entries)
		return mat->entries[(mat->rows * colum) + row];
	//else
		//exit(1);
}

// get the adress of an element
inline double* get_adr_element(const struct Matrix* mat, size_t row, size_t colum){
	return &mat->entries[(mat->rows * colum) + row];
}

// set a single element from the matrix based on its row and colum by the given value
inline void set_element(const struct Matrix* mat, size_t row, size_t colum, double value){
	mat->entries[(mat->rows * colum) + row] = value;
}

inline double rand_double(void){
	return (2 * ((double) rand())) / ((double) RAND_MAX) - 1.0l;
}

// free allocated memory for the matrix
inline void free_mat(struct Matrix* mat){
	mat->colums = 0;
	mat->rows = 0;
	free(mat->entries);
	mat->entries = NULL;
}

inline void print_mat(const struct Matrix* mat){
	for(size_t i = 0; i != mat->colums; i++){
		printf("[");
		for(size_t j = 0; j != mat->rows; j++){
			printf("%lf, \t", get_element(mat, j, i));
		}
		printf("]\n");
	}
	printf("\n");
}



// src matrix should be unintializd
// otherwise it will get overwritten
inline void cpy_mat(struct Matrix* dest, const struct Matrix* src){
	// initialize the dest matrix (its should be uninitialized before this function)
	init_mat(dest, src->rows, src->colums);
	// copy the data from src matrix to the dest matrix
	memcpy(dest->entries, src->entries, sizeof(src->entries[0]) * src->rows * src->colums);
}

inline void randomize_mat(struct Matrix* mat){
	srand(time(NULL));
	for(size_t i = 0; i != mat->colums; i++)
		for(size_t j = 0; j != mat->rows; j++)
			set_element(mat, j, i, rand_double());
}

inline void square_mat(struct Matrix* mat){
	for(size_t i = 0; i != mat->colums; i++){
		for(size_t j = 0; j != mat->rows; j++){
			double crntelem = get_element(mat, j, i);
			set_element(mat, j, i, crntelem * crntelem);
		}
	}
}

// segmoid mathematical function
inline double segmoid(double num){
	return 1/(1 + expl(-num));
}


// apply the segmoid function on a matrix
inline struct Matrix segmoid_mat(struct Matrix* mat){
	struct Matrix result;
	init_mat(&result, mat->rows, mat->colums);
	for (size_t i = 0; i != mat->rows; i++){
		for (size_t j = 0; j != mat->colums; j++){
			double crnt = get_element(mat, i, j);
			set_element(&result, i, j, segmoid(crnt));
		}
	}
	return result;
}

// WARNIING: this works only with pointers
#define check_dimensions(mat1, mat2) ((mat1->rows) == (mat2->rows) && (mat1->colums) == (mat2->colums))

// matrix dot product
inline struct Matrix dot_mat_mat(const struct Matrix* mat1, const struct Matrix* mat2){
	if(mat1->colums == mat2->rows){
		struct Matrix result;
		init_mat(&result, mat1->rows, mat2->colums);

		for(size_t i = 0; i != mat1->rows; i++){
			for(size_t j = 0; j != mat2->colums; j++){

				double sum = 0;
				for(size_t k = 0; k != mat2->rows; k++)
					sum += get_element(mat1, i, k) * get_element(mat2, k, j);

				set_element(&result, i, j, sum);
			}
		}

		return result;

	} else {
		fprintf(stderr, "Matrix dimentions mismatch: %zux%zu . %zux%zu\n",
		                       mat1->rows, mat1->colums, mat2->rows, mat2->colums);
		assert(1);
	}
}

// matrix by matrix multiplication
inline struct Matrix mul_mat_mat(const struct Matrix* mat1, const struct Matrix* mat2) {
	if (check_dimensions(mat1, mat2)){
		struct Matrix result;
		init_mat(&result, mat1->rows, mat1->colums);
		for (size_t i = 0; i != mat1->rows; i++)
			for (size_t j = 0; j != mat2->colums; j++)
				set_element(&result, i, j,
				     (get_element(mat1, i, j) * get_element(mat2, i, j)));
		return result;
	} else {
		printf("Dimension mistmatch multiply: %zux%zu * %zux%zu\n",
		       mat1->rows, mat1->colums, mat2->rows, mat2->colums);
		assert(1);
	}
}

// matrix by scalar multiplication
inline void mul_mat_scalar(const struct Matrix* mat, const double number) {
	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			set_element(mat, i, j, (get_element(mat, i, j) * number));
}

// matrix by scalar deviding
inline void dev_mat_scalar(const struct Matrix* mat, const double number) {
	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			set_element(mat, i, j, (get_element(mat, i, j) / number));
}

// matrix addition
inline struct Matrix add_mat_mat(const struct Matrix* mat1, const struct Matrix* mat2){
	if (check_dimensions(mat1, mat2)) {
		struct Matrix result;
		init_mat(&result, mat1->rows, mat1->colums);
		for (size_t i = 0; i != mat1->rows; i++)
			for (size_t j = 0; j != mat2->colums; j++)
				set_element(&result, i, j, 
				           (get_element(mat1, i, j) + get_element(mat2, i, j)));
		return result;
	} else {
		printf("Dimension mistmatch add: %zux%zu + %zux%zu\n",
		       mat1->rows, mat1->colums, mat2->rows, mat2->colums);
		assert(1);
	}
}

inline struct Matrix add_mat_scalar(const struct Matrix* mat, const double number) {
	struct Matrix result;
	init_mat(&result, mat->rows, mat->colums);

	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			set_element(&result, i, j, get_element(mat, i, j) + number);

	return result;
}

// matrix subtraction
inline struct Matrix sub_mat_mat(const struct Matrix* mat1, const struct Matrix* mat2){
	if (check_dimensions(mat1, mat2)) {
		struct Matrix result;
		init_mat(&result, mat1->rows, mat1->colums);
		for (size_t i = 0; i != mat1->rows; i++)
			for (size_t j = 0; j != mat2->colums; j++)
				set_element(&result, i, j, 
				           (get_element(mat1, i, j) - get_element(mat2, i, j)));
		return result;
	} else {
		printf("Dimension mistmatch subtract: %zux%zu - %zux%zu\n",
		       mat1->rows, mat1->colums, mat2->rows, mat2->colums);
		assert(1);
	}
}

// matrix transposition
inline struct Matrix trans_mat(const struct Matrix* mat){
	struct Matrix result;
	init_mat(&result, mat->colums, mat->rows);

	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			set_element(&result, j, i, get_element(mat, i, j));

	return result;
}

// apply the derivative of the segmoid finction to a matrix
inline struct Matrix sigmoidPrime(struct Matrix* mat) {
	struct Matrix ones;
	init_mat(&ones, mat->rows, mat->colums);
	fill_mat(&ones, 1.0);
	struct Matrix subtracted = sub_mat_mat(&ones, mat);
	struct Matrix multiplied = mul_mat_mat(mat, &subtracted);
	free_mat(&ones);
	free_mat(&subtracted);
	return multiplied;
}

// flaten matrix (IDK whats that, I escaped linear algebra class)
inline struct Matrix flatten_mat(struct Matrix* mat, bool axis) {
	// Axis = 0 -> raw Vector, Axis = 1 -> colum Vector
	struct Matrix result;
	cpy_mat(&result, mat);
	if(axis){
		result.rows *= result.colums; 
		result.colums = 1;
	}else{
		result.colums *= result.rows;
		result.rows =1;
	}
	return result;
}

// IDK what this does
inline struct Matrix softmax(struct Matrix* mat) {
	double sum = 0;
	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			sum += exp(get_element(mat, i, j));

	struct Matrix result;
	init_mat(&result, mat->rows, mat->colums);
	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			set_element(&result, i, j, exp(get_element(mat, i, j)) / sum);

	return result;
}

inline size_t matrix_argmax(struct Matrix* mat){
	// Expects a Mx1 matrix
	double max_score = 0;
	size_t max_idx = 0;
	for (size_t i = 0; i != mat->rows; i++){
		if (get_element(mat, i, 0) > max_score){
			max_score = get_element(mat, i, 0);
			max_idx = i;
		}
	}
	return max_idx;
}

#endif