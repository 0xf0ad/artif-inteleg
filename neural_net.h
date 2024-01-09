#ifndef NN_H_
#define NN_H_

#include "csv_parsser.h"
#include "matrix.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

struct NeuralNet{
	uint32_t num_layers;
	size_t* layer_sizes;
	size_t num_weights;
	// NOTE : that num_weights doesnt indicate the size of the array
	// it`s size is (num_layers - 1)
	struct Matrix* weights;
	double learning_rate;
};


void init_net(struct NeuralNet* net, size_t* p_layer_sizes, uint32_t p_num_layers, double p_learning_rate){
	net->learning_rate = p_learning_rate;
	net->num_layers = p_num_layers;
	net->layer_sizes = (size_t*) malloc(sizeof(size_t) * p_num_layers);
	memcpy(net->layer_sizes, p_layer_sizes, p_num_layers);

	size_t num_weights = 0;
	for(uint32_t i = 1; i != p_num_layers; i++)
		num_weights += p_layer_sizes[i] * p_layer_sizes[i-1];

	net->num_weights = num_weights;
	net->weights = (struct Matrix*) malloc(sizeof(struct Matrix) * (p_num_layers - 1));
	for(uint32_t i = 0; i != (p_num_layers - 1); i++){
		// encouter for the biasses therfor we add one row and one colum for them
		init_mat(&net->weights[i], p_layer_sizes[i+1] + 1 , p_layer_sizes[i] + 1);
		randomize_mat(&net->weights[i]);
	}
}

void free_net(struct NeuralNet* net){
	net->num_layers = 0;
	free(net->layer_sizes);
	net->layer_sizes = NULL;

	for(size_t i = 0; i != net->num_layers; i++)
		free_mat(&net->weights[i]);

	net->num_weights = 0;
	free(net->weights);
	net->weights = NULL;
	net->learning_rate = 0.0l;
}

inline void network_train(struct NeuralNet* net, struct Matrix* in_mat, struct Matrix* out_mat){
	// feed-forward
	uint32_t num_layers = net->num_layers;
	struct Matrix in_layers[num_layers - 1];
	struct Matrix activations[num_layers];
	activations[0] = *in_mat;
	// dot every matrix layer with the matrix of the prev layer exept for the input layer
	// and apply the segmoid function to obtain the output
	for(size_t i = 0; i != (num_layers - 1); i++){
		in_layers[i] = dot_mat_mat(&net->weights[i], &activations[i]);
		//in_layers[i] = add_mat_scalar(&tmp, net->biases[i]);
		activations[i+1] = segmoid_mat(&in_layers[i]);
	}

	// errors
	struct Matrix errors[num_layers];
	errors[num_layers-1] = sub_mat_mat(out_mat, &activations[num_layers - 1]);
	// calculat error for each layer backward
	for(uint32_t i = (num_layers - 2); i != 0; i--){
		struct Matrix transposed_mat = trans_mat(&net->weights[i]);
		errors[i] = dot_mat_mat(&transposed_mat, &errors[i+1]);
		free_mat(&transposed_mat);
	}

	for(uint32_t i = (num_layers - 1); i != 0; i--){
		struct Matrix sigmoid_primed_mat = sigmoidPrime(&activations[i]);
		struct Matrix multiplied_mat = mul_mat_mat(&errors[i], &sigmoid_primed_mat);
		struct Matrix transposed_mat = trans_mat(&activations[i-1]);
		struct Matrix dot_mat = dot_mat_mat(&multiplied_mat, &transposed_mat);
		mul_mat_scalar(&dot_mat ,net->learning_rate);
		struct Matrix added_mat = add_mat_mat(&net->weights[i-1], &dot_mat);

		free_mat(&net->weights[i-1]); // Free the old weights before replacing
		net->weights[i-1] = added_mat;

		free_mat(&sigmoid_primed_mat);
		free_mat(&multiplied_mat);
		free_mat(&transposed_mat);
		free_mat(&dot_mat);
	}
}

void network_train_batch_imgs(struct NeuralNet* net, struct Data* data, size_t batch_size){
	for (size_t i = 0; i != batch_size; i++){
		printf("image No: %zu\n", i);
		struct Data* cur_data = &data[i];
		struct Matrix output;
		// 10 outputs and a bias its just easier to add biases to all layers
		init_mat(&output, 11, 1);
		fill_mat(&output, 0.0l);
		set_element(&output, cur_data->label, 0, 1);
		network_train(net, &cur_data->mat, &output);
		free_mat(&output);
	}
}

inline struct Matrix network_predict(struct NeuralNet* net, struct Matrix* input_data){
	struct Matrix inputs[net->num_layers];
	struct Matrix outputs[net->num_layers];
	outputs[0] = *input_data;

	for(size_t i = 1; i != net->num_layers; i++){
		inputs[i] = dot_mat_mat(&net->weights[i-1], &outputs[i-1]);
		outputs[i] = segmoid_mat(&inputs[i]);
	}

	struct Matrix result = softmax(&outputs[net->num_layers - 1]);

	for(size_t i = 1; i != net->num_layers; i++){
		free_mat(&inputs[i]);
		free_mat(&outputs[i]);
	}
	return result;
}

struct Matrix network_predict_img(struct NeuralNet* net, struct Data* data){
	struct Matrix img_data = flatten_mat(&data->mat, 1);
	struct Matrix result = network_predict(net, &img_data);
	free_mat(&img_data);
	return result;
}

double network_predict_imgs(struct NeuralNet* net, struct Data* data, uint32_t n){
	uint32_t n_correct = 0;
	for (uint32_t i = 0; i != n; i++){
		struct Matrix prediction = network_predict_img(net, &data[i]);
		printf("JABHA %zu hiya rah %d\n", matrix_argmax(&prediction), data[i].label);
		if (matrix_argmax(&prediction) == data[i].label){
			n_correct++;
		}

		free_mat(&prediction);
	}
	return 1.0 * n_correct / n;
}

#endif // NN_H_
