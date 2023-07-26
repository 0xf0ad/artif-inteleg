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
		printf("mat %p, size %zu x %zu\n", &net->weights[i], net->weights[i].rows, net->weights[i].colums);
		randomize_mat(&net->weights[i]);
	}
	printf("allocated %d matrices, on %p\n", p_num_layers - 1, net->weights);
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

/*void network_train(struct NeuralNet* net, struct Matrix* input, struct Matrix* output){	
	
	// Feed forward
	struct Matrix hidden_inputs	= dot_mat_mat(&net->weights[0], input);
	struct Matrix hidden_outputs = segmoid_mat(&hidden_inputs);
	struct Matrix final_inputs = dot_mat_mat(&net->weights[1], &hidden_outputs);
	struct Matrix final_outputs = segmoid_mat(&final_inputs);

	// Find errors
	struct Matrix output_errors = sub_mat_mat(output, &final_outputs);
	struct Matrix transposed_mat = trans_mat(&net->weights[1]);
	struct Matrix hidden_errors = dot_mat_mat(&transposed_mat, &output_errors);
	free_mat(&transposed_mat);

	// Backpropogate
	// output_weights = add(
	//		 output_weights, 
	//     scale(
	// 			  net->lr, 
	//			  dot(
	// 		 			multiply(
	// 						output_errors, 
	//				  	sigmoidPrime(final_outputs)
	//					), 
	//					transpose(hidden_outputs)
	// 				)
	//		 )
	// )
	struct Matrix sigmoid_primed_mat = sigmoidPrime(&final_outputs);
	struct Matrix multiplied_mat = mul_mat_mat(&output_errors, &sigmoid_primed_mat);
	transposed_mat = trans_mat(&hidden_outputs);
	struct Matrix dot_mat = dot_mat_mat(&multiplied_mat, &transposed_mat);
	mul_mat_scalar(&dot_mat, net->learning_rate);
	struct Matrix added_mat = add_mat_mat(&net->weights[0], &dot_mat);
	//matrix_print(added_mat);

	free_mat(&net->weights[0]); // Free the old weights before replacing
	net->weights[0] = added_mat;

	free_mat(&sigmoid_primed_mat);
	free_mat(&multiplied_mat);
	free_mat(&transposed_mat);
	free_mat(&dot_mat);

	// hidden_weights = add(
	// 	 net->hidden_weights,
	// 	 scale (
	//			net->learning_rate
	//    	dot (
	//				multiply(
	//					hidden_errors,
	//					sigmoidPrime(hidden_outputs)	
	//				)
	//				transpose(inputs)
	//      )
	// 	 )
	// )
	// Reusing variables after freeing memory
	sigmoid_primed_mat = sigmoidPrime(&hidden_outputs);
	multiplied_mat = mul_mat_mat(&hidden_errors, &sigmoid_primed_mat);
	transposed_mat = trans_mat(input);
	dot_mat = dot_mat_mat(&multiplied_mat, &transposed_mat);
	mul_mat_scalar(&dot_mat, net->learning_rate);
	added_mat = add_mat_mat(&net->weights[1], &dot_mat);
	free_mat(&net->weights[1]); // Free the old hidden_weights before replacement
	net->weights[1] = added_mat;
	//matrix_print(dot_mat);

	free_mat(&sigmoid_primed_mat);
	free_mat(&multiplied_mat);
	free_mat(&transposed_mat);
	free_mat(&dot_mat);

	// Free matrices
	free_mat(&hidden_inputs);
	free_mat(&hidden_outputs);
	free_mat(&final_inputs);
	free_mat(&final_outputs);
	free_mat(&output_errors);
	free_mat(&hidden_errors);
}*/

/*#define PIXEL_SCALE(x) (((float)(x)) / 255.0f)

void feed_forward(struct NeuralNet* net, struct Matrix* image, float* activations, uint32_t num_output_nodes){
    int i, j;

    for (i = 0; i != num_output_nodes; i++) {
        activations[i] = net->biases[i];

        for (j = 0; j != (image->rows * image->colums); j++) {
            activations[i] += net->weights[i][j] * PIXEL_SCALE(image->entries[j]);
							  get_element(net->weights, i, j)
        }
    }

    neural_network_softmax(activations, num_output_nodes);
}*/

void network_train(struct NeuralNet* net, struct Matrix* in_mat, struct Matrix* out_mat){
	// feed-forward
	//print_mat(in_mat);
	uint32_t num_layers = net->num_layers;
	struct Matrix in_layers[num_layers - 1];
	struct Matrix activations[num_layers];
	struct Matrix* output_layer = &activations[num_layers - 1];
	activations[0] = *in_mat;
	// dot every matrix layer with the matrix of the prev layer exept for the input layer
	// and apply the segmoid function to obtain the output
	for(size_t i = 0; i != (num_layers - 1); i++){
		//printf("entries: %p\tweights : %p\n", net->weights[i].entries, net->weights);
		in_layers[i] = dot_mat_mat(&net->weights[i], &activations[i]);
		//printf("%zu x %zu . %zu x %zu = %zu x %zu\n", net->weights[i].rows, net->weights[i].colums, activations[i].rows, activations[i].colums, in_layers[i].rows, in_layers[i].colums);
		//in_layers[i] = add_mat_scalar(&tmp, net->biases[i]);
		print_mat(&in_layers[i]);
		activations[i+1] = segmoid_mat(&in_layers[i]);
		print_mat(&activations[i+1]);
		printf("\n######################\n# this is the %zu time #\n######################\n", i);
	}
	//print_mat(&activations[num_layers-1]);

	// errors
	struct Matrix errors[num_layers];
	printf("matrix of size : %zu x %zu\n", activations[num_layers - 1].rows, activations[num_layers - 1].colums);
	print_mat(output_layer);
	print_mat(out_mat);
	struct Matrix cost = sub_mat_mat(output_layer, out_mat);
	square_mat(&cost);
	//init_mat(&cost, out_mat->rows, out_mat->colums);
	double v_cost = 0.0l;
	double tmp;
	/*for (size_t i = 0; i != output_layer->rows * output_layer->rows; i++){
		tmp = get_element(output_layer, i, 0) - get_element(out_mat, i, 0);
		set_element(&cost, 0, i, tmp * tmp);
		v_cost += tmp * tmp;
	}*/
	print_mat(&cost);
	//v_cost /= (activations->rows * activations->colums);
	// set the output error
	//printf("entries: %p\tweights : %p\n", net->weights->entries, net->weights);
	errors[num_layers-1] = sub_mat_mat(out_mat, output_layer);
	//print_mat(&errors[num_layers - 1]);
	// calculat error for each layer backward
	for(uint32_t i = (num_layers - 2); i != 0; i--){
		//printf("second %d entries: %p\tweights : %p\n", i, net->weights[i].entries, net->weights);
		struct Matrix transposed_mat = trans_mat(&net->weights[i]);
		errors[i] = dot_mat_mat(&transposed_mat, &errors[i+1]);
		//print_mat(&errors[i]);
		free_mat(&transposed_mat);
	}

	for(uint32_t i = (num_layers - 1); i != 0; i--){
		struct Matrix sigmoid_primed_mat = sigmoidPrime(&activations[i]);
		struct Matrix multiplied_mat = mul_mat_mat(&errors[i], &sigmoid_primed_mat);
		struct Matrix transposed_mat = trans_mat(&activations[i-1]);
		struct Matrix dot_mat = dot_mat_mat(&multiplied_mat, &transposed_mat);
		//print_mat(&dot_mat);
		mul_mat_scalar(&dot_mat ,net->learning_rate);
		struct Matrix added_mat = add_mat_mat(&net->weights[i-1], &dot_mat);
		//print_mat(&dot_mat);

		printf("%f and %f tha error : %f\n", net->weights[i-1].entries[0], added_mat.entries[0], cost);
		free_mat(&net->weights[i-1]); // Free the old weights before replacing
		net->weights[i-1] = added_mat;
		//printf("\n");
		//print_mat(&net->weights[i-1]);

		free_mat(&sigmoid_primed_mat);
		free_mat(&multiplied_mat);
		free_mat(&transposed_mat);
		free_mat(&dot_mat);
	}

	//printf("OUT OF OF THE LOOP\n");

	/*free_mat(&sigmoid_primed_mat);
	free_mat(&multiplied_mat);

	for(uint32_t i = 0; i != num_layers; i++){
		free_mat(&out_layers[i]);
		free_mat(&errors[i]);
		
		if(i != (num_layers - 1))
			free_mat(&in_layers[i]);
	}*/

}

void network_train_batch_imgs(struct NeuralNet* net, struct Data* data, size_t batch_size){
	for (size_t i = 0; i != batch_size; i++){
		if (!(i % 100))
			printf("image No: %zu\n", i);
		struct Data* cur_data = &data[i];
		//struct Matrix img_data = flatten_mat(&cur_data->mat, 1);
		//print_mat(&cur_data->mat);
		struct Matrix output;
		// 10 outputs and a bias its just easier to add biases to all layers
		init_mat(&output, 11, 1);
		fill_mat(&output, 0.0l);
		set_element(&output, cur_data->label, 0, 1);
		network_train(net, &cur_data->mat, &output);
		//print_mat(&cur_data->mat);
		free_mat(&output);
		free_mat(&cur_data->mat);
	}
}

struct Matrix network_predict(struct NeuralNet* net, struct Matrix* input_data){
	struct Matrix inputs[net->num_layers];
	struct Matrix outputs[net->num_layers];
	outputs[0] = *input_data;

	for(size_t i = 1; i != net->num_layers; i++){
		inputs[i] = dot_mat_mat(&net->weights[i-1], &outputs[i-1]);
		print_mat(&inputs[i]);
		outputs[i] = segmoid_mat(&inputs[i]);
	}

	struct Matrix result = softmax(&outputs[net->num_layers - 1]);
	//print_mat(&outputs[net->num_layers - 1]);
	print_mat(&result);

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

double network_predict_imgs(struct NeuralNet* net, struct Data* data, uint32_t n) {
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