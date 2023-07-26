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
void init_mat(struct Matrix* mat, size_t rows, size_t colums){
	mat->rows = rows;
	mat->colums = colums;
	mat->entries = (double*) malloc(sizeof(double) * rows * colums);
}

// fill the entire matrix with one single value
void fill_mat(struct Matrix* mat, double val){
	size_t iters = mat->rows * mat->colums;
	for(size_t i = 0; i != iters; i++)
		mat->entries[i] = val;
}

// get a single element from the matrix based on its row and colum
double get_element(const struct Matrix* mat, size_t row, size_t colum){
	//if(mat->entries)
		return mat->entries[(mat->rows * colum) + row];
	//else
		//exit(1);
}

// get the adress of an element
double* get_adr_element(const struct Matrix* mat, size_t row, size_t colum){
	return &mat->entries[(mat->rows * colum) + row];
}

// set a single element from the matrix based on its row and colum by the given value
void set_element(const struct Matrix* mat, size_t row, size_t colum, double value){
	mat->entries[(mat->rows * colum) + row] = value;
}

double rand_double(void){
	return ((double) rand()) / ((double) RAND_MAX);
}

// free allocated memory for the matrix
void free_mat(struct Matrix* mat){
	mat->colums = 0;
	mat->rows = 0;
	free(mat->entries);
	mat->entries = NULL;
}

void print_mat(const struct Matrix* mat){
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
void cpy_mat(struct Matrix* dest, const struct Matrix* src){
	// initialize the dest matrix (its should be uninitialized before this function)
	init_mat(dest, src->rows, src->colums);
	// copy the data from src matrix to the dest matrix
	memcpy(dest->entries, src->entries, (src->rows * src->colums));
}

void randomize_mat(struct Matrix* mat){
	srand(time(NULL));
	fill_mat(mat, 0.0);
	/*for(size_t i = 0; i != mat->colums; i++)
		for(size_t j = 0; j != mat->rows; j++)
			set_element(mat, j, i, rand_double());*/
}

struct Matrix square_mat(struct Matrix* mat){
	for(size_t i = 0; i != mat->colums; i++)
		for(size_t j = 0; j != mat->rows; j++){
			double crntelem = get_element(mat, j, i);
			set_element(mat, j, i, crntelem * crntelem);
		}
}

// segmoid mathematical function
double segmoid(double num){
	return 1.0l/(1.0l + expl(-num));
}


// apply the segmoid function on a matrix
struct Matrix segmoid_mat(struct Matrix* mat){
	struct Matrix result;
	cpy_mat(&result, mat);
	for (size_t i = 0; i != result.rows; i++)
		for (size_t j = 0; j != result.colums; j++){
			double* crnt = get_adr_element(&result, i, j);
			if(!crnt)
				assert(1);
			*crnt = segmoid(*crnt);
		}
	return result;
}

// WARNIING: this works only with pointers
#define check_dimensions(mat1, mat2) ((mat1->rows) == (mat2->rows) && (mat1->colums) == (mat2->colums))

// matrix dot product
struct Matrix dot_mat_mat(const struct Matrix* mat1, const struct Matrix* mat2){
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

struct Matrix dot(struct Matrix *m1, struct Matrix *m2) {
	if (m1->colums == m2->rows) {
		struct Matrix m ; init_mat(&m, m1->rows, m2->colums);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->colums; j++) {
				double sum = 0;
				for (int k = 0; k < m2->rows; k++) {
					sum += get_element(m1, i, k) * get_element(m2, k, j);
				}
				set_element(&m, i, j, sum);
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->colums, m2->rows, m2->colums);
		exit(1);
	}
}

// matrix by matrix multiplication
struct Matrix mul_mat_mat(const struct Matrix* mat1, const struct Matrix* mat2) {
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
void mul_mat_scalar(const struct Matrix* mat, const double number) {
	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			set_element(mat, i, j, (get_element(mat, i, j) * number));
}

// matrix by scalar deviding
void dev_mat_scalar(const struct Matrix* mat, const double number) {
	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			set_element(mat, i, j, (get_element(mat, i, j) / number));
}

// matrix addition
struct Matrix add_mat_mat(const struct Matrix* mat1, const struct Matrix* mat2){
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

struct Matrix add_mat_scalar(const struct Matrix* mat, const double number) {
	struct Matrix result;
	init_mat(&result, mat->rows, mat->colums);

	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			set_element(&result, i, j, get_element(mat, i, j) + number);

	return result;
}

// matrix subtraction
struct Matrix sub_mat_mat(const struct Matrix* mat1, const struct Matrix* mat2){
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
struct Matrix trans_mat(const struct Matrix* mat){
	struct Matrix result;
	init_mat(&result, mat->colums, mat->rows);

	for (size_t i = 0; i != mat->rows; i++)
		for (size_t j = 0; j != mat->colums; j++)
			set_element(&result, j, i, get_element(mat, i, j));

	return result;
}

// apply the derivative of the segmoid finction to a matrix
struct Matrix sigmoidPrime(struct Matrix* mat) {
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
struct Matrix flatten_mat(struct Matrix* mat, bool axis) {
	// Axis = 0 -> raw Vector, Axis = 1 -> colum Vector
	struct Matrix result;
	//init_mat(&result, mat->rows, mat->colums);
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
struct Matrix softmax(struct Matrix* mat) {
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

size_t matrix_argmax(struct Matrix* mat){
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