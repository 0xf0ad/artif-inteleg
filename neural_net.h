#ifndef NN_H_
#define NN_H_

#include "csv_parsser.h"
#include "matrix.h"
#include <stdint.h>

typedef struct{
	uint32_t num_layers;
	size_t* layer_sizes;
	size_t num_weights;
	// NOTE : that num_weights doesnt indicate the size of the weights array
	// it`s actually size is (num_layers - 1)
	matrix_t* bias;
	matrix_t* weights;
	double learning_rate;
} neuralnet_t;


void init_net(neuralnet_t* net, size_t* p_layer_sizes, uint32_t p_num_layers, double p_learning_rate);
void free_net(neuralnet_t* net);
void network_train(neuralnet_t* net, matrix_t* in_mat, matrix_t* out_mat);
void network_train_batch_imgs(neuralnet_t* net, data_t* data, size_t batch_size);
matrix_t network_predict(neuralnet_t* net, matrix_t* input_data);
matrix_t network_predict_img(neuralnet_t* net, data_t* data);
double network_predict_imgs(neuralnet_t* net, data_t* data, uint32_t n);


#endif // NN_H_
