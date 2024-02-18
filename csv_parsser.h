#ifndef CSV_PARSS_H_
#define CSV_PARSS_H_

#include "matrix.h"

// the number of nodes in the input layer
// it is fixed because I will be trainning only the MNIST set
// witch compose of 28x28 pixels = 628 pixels
#define IMG_SIZE     28
#define INPUT_NODES  IMG_SIZE * IMG_SIZE // 784

typedef struct{
	uint8_t label;
	matrix_t mat;
}data_t;

data_t* csv_parss(const char* filename, size_t* p_num_objects);

#endif
