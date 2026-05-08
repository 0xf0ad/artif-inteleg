#include "matrix.h"
#include "neural_net.h"
#include "csv_parsser.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void init_net(neuralnet_t* net, size_t* p_layer_sizes, uint32_t p_num_layers, double p_learning_rate){
	net->learning_rate = p_learning_rate;
	net->num_layers = p_num_layers;
	net->layer_sizes = (size_t*) malloc(sizeof(size_t) * p_num_layers);
	memcpy(net->layer_sizes, p_layer_sizes, sizeof(size_t) * p_num_layers);

	size_t num_weights = 0;
	for(uint32_t i = 1; i != p_num_layers; i++)
		num_weights += p_layer_sizes[i] * p_layer_sizes[i-1];
	net->num_weights = num_weights;

	net->bias    = (matrix_t*) malloc(sizeof(matrix_t) * (p_num_layers - 1));
	net->weights = (matrix_t*) malloc(sizeof(matrix_t) * (p_num_layers - 1));
	srand(time(NULL));
	for(uint32_t i = 0; i < (p_num_layers - 1); i++){
		init_mat(&net->bias[i], p_layer_sizes[i+1], 1);
		fill_mat(&net->bias[i], 0.0); // Initialize biases to zero

		init_mat(&net->weights[i], p_layer_sizes[i+1], p_layer_sizes[i]);
		// Use Xavier initialization for weights
		for(size_t j = 0; j < net->weights[i].rows; j++){
			for(size_t k = 0; k < net->weights[i].colums; k++){
				set_element(&net->weights[i], j, k, 
				           rand_xavier(p_layer_sizes[i], p_layer_sizes[i+1]));
			}
		}
	}
}

void free_net(neuralnet_t* net){
	free(net->layer_sizes);
	net->layer_sizes = NULL;

	for(size_t i = 0; i < net->num_layers - 1; i++){
		free_mat(&net->weights[i]);
		free_mat(&net->bias[i]);
	}
	net->num_layers = 0;

	free(net->weights);
	net->num_weights = 0;
	net->weights = NULL;
	net->learning_rate = 0.l;
}

void network_train(neuralnet_t* net, matrix_t* in_mat, matrix_t* out_mat){
	// feed-forward
	uint32_t num_layers = net->num_layers;
	matrix_t in_layers[num_layers - 1];
	matrix_t activations[num_layers];
	cpy_mat(&activations[0], in_mat);
	// dot every matrix layer with the matrix of the prev layer exept for the input layer
	// and apply the segmoid function to obtain the output
	for(size_t i = 0; i < (num_layers - 1); i++){
		matrix_t dots = dot_mat_mat(&net->weights[i], &activations[i]);
		in_layers[i] = add_mat_mat(&net->bias[i], &dots);
		activations[i+1] = segmoid_mat(&in_layers[i]);
		free_mat(&dots);
	}

	// errors
	matrix_t errors[num_layers];
	errors[num_layers-1] = sub_mat_mat(out_mat, &activations[num_layers - 1]);
	// calculat error for each layer backward
	for(int32_t i = (num_layers - 2); i > 0; i--){
		matrix_t transposed_mat = trans_mat(&net->weights[i]);
		errors[i] = dot_mat_mat(&transposed_mat, &errors[i+1]);
		free_mat(&transposed_mat);
	}

	for(int32_t i = (num_layers - 1); i > 0; i--){
		matrix_t sigmoid_primed_act = sigmoidPrime(&activations[i]);
		matrix_t delta = mul_mat_mat(&errors[i], &sigmoid_primed_act);

		// Update bias: bias = bias + learning_rate * delta
		matrix_t scaled_delta = delta;
		mul_mat_scalar(&scaled_delta, net->learning_rate);
		matrix_t new_bias = add_mat_mat(&net->bias[i-1], &scaled_delta);
		free_mat(&net->bias[i-1]);
		net->bias[i-1] = new_bias;

		// Update weights: weights = weights + learning_rate * (delta * activation^T)
		matrix_t transposed_act = trans_mat(&activations[i-1]);
		matrix_t new_weights = dot_mat_mat(&delta, &transposed_act);
		mul_mat_scalar(&new_weights, net->learning_rate);
		matrix_t added_mat = add_mat_mat(&net->weights[i-1], &new_weights);

		free_mat(&net->weights[i-1]);
		net->weights[i-1] = added_mat;

		free_mat(&sigmoid_primed_act);
		free_mat(&delta);
		free_mat(&transposed_act);
		free_mat(&new_weights);
	}

	for(uint32_t i = 0; i < num_layers - 1; i++){
		free_mat(&in_layers[i]);
		free_mat(&activations[i]);
		free_mat(&errors[i+1]);
	}
	free_mat(&activations[num_layers-1]);
}

void network_train_batch_imgs(neuralnet_t* net, data_t* data, size_t batch_size){
	for (size_t i = 0; i != batch_size; i++){
		//printf("training on image No: %zu   %.2f%%\r", i+1, ((float)(i+1) / (float)batch_size) * 100.f);
		data_t* cur_data = &data[i];
		matrix_t output;
		init_mat(&output, 10, 1);
		fill_mat(&output, 0.0l);
		set_element(&output, cur_data->label, 0, 1);
		network_train(net, &cur_data->mat, &output);
		free_mat(&output);
	}
}

matrix_t network_predict(neuralnet_t* net, matrix_t* input_data){
	matrix_t inputs[net->num_layers];
	matrix_t outputs[net->num_layers];
	// the following line is better left this way
	outputs[0] = *input_data;

	for(size_t i = 1; i < net->num_layers; i++){
		matrix_t dots = dot_mat_mat(&net->weights[i-1], &outputs[i-1]);
		inputs[i] = add_mat_mat(&net->bias[i-1], &dots);
		outputs[i] = segmoid_mat(&inputs[i]);
		free_mat(&dots);
	}

	matrix_t result = softmax(&outputs[net->num_layers - 1]);

	for(size_t i = 1; i < net->num_layers; i++){
		free_mat(&inputs[i]);
		free_mat(&outputs[i]);
	}
	return result;
}

matrix_t network_predict_img(neuralnet_t* net, data_t* data){
	matrix_t img_data = flatten_mat(&data->mat, 1);
	matrix_t result = network_predict(net, &img_data);
	free_mat(&img_data);
	return result;
}

double network_predict_imgs(neuralnet_t* net, data_t* data, uint32_t n){
	uint32_t n_correct = 0;
	for (uint32_t i = 0; i < n; i++){
		matrix_t prediction = network_predict_img(net, &data[i]);
		n_correct += matrix_argmax(&prediction) == data[i].label;
		free_mat(&prediction);
	}
	return ((float)n_correct) / n;
}

// returns an array which the first element is the number of line in a file
// and other elements are the lenght of lines by order (the i-th line lengh is the (i+1)-th element of the array)
size_t count_lines(FILE* file){
	size_t counter = 0;

	while(!feof_unlocked(file))
		if(fgetc_unlocked(file) == '\n')
			counter++;

	// clear the EOF reacheed by the prev loop
	clearerr_unlocked(file);
	rewind(file);
	return counter;
}

// WARNNING : the char** that will b returned is heap-allocated so you should free it
// and also all the strings who is pointed to
const char** extract_lines(const char* filename, size_t* num_lines){
	FILE* file = fopen(filename, "r");

	if(!file){
		fprintf(stderr, "ERROR: file %s had failed to open, does it even existe ?\n", filename);
		return NULL;
	}

	size_t file_num_lines = count_lines(file);
	(*num_lines) = file_num_lines;

	const char** str = (const char**) malloc(sizeof(char*) * file_num_lines);
	if(!str){
		fprintf(stderr, "ERROR: da hell r u running this on ?\n");
		fprintf(stderr, "ERROR: we didnt start yet and r already running out of memory\n");
		fprintf(stderr, "ERROR: bro buy yourself new potatos\n");
		fclose(file);
		return NULL;
	}

	size_t num_chars[file_num_lines];
	memset(num_chars, 0, (sizeof(size_t) * file_num_lines));
	for(size_t i = 0; i != file_num_lines; i++){
		// this is a temporary string
		char* tmpstr = NULL;
		
		if(getline(&tmpstr, &num_chars[i], file) == -1){
			fprintf(stderr, "ERROR: bro is that a file ?\n");
			// if this failed I am not going to free this junk
			// insted I will exit and let the OS reclaim its emory
			// its not like I will overpower somecapacitor in that stick of yours
			// supposing you dont use windows
			assert(1);
		} else {
			tmpstr[ strcspn(tmpstr, "\n") ] = '\0';
			str[i] = strdup(tmpstr);
			free(tmpstr);
		}
	}

	fclose(file);
	return str;
}

data_t* csv_parss(const char* filename, size_t* p_num_objects){
	const char** strings = extract_lines(filename, p_num_objects);

	data_t* data = (data_t*) malloc(sizeof(data_t) * (*p_num_objects));
	for(size_t i = 0; i != (*p_num_objects); i++){
		// we assume the label is always a single digit so one character
		// so we convert a single character
		data[i].label = strings[i][0] - '0';
		// we expect consumming a char for the label and an other for the comma ,
		const char* string_offset = (strings[i] + 2);
		init_mat(&data[i].mat, INPUT_NODES, 1);
		for(size_t j = 0; j < INPUT_NODES; j++){
			sscanf(string_offset, "%lf", &data[i].mat.entries[j]);
			// we add a one to encounter the comma character
			string_offset += strcspn(string_offset, ",") + 1;
		}
		// Normalize pixel values to 0-1 range by dividing by 255
		dev_mat_scalar(&data[i].mat, 255.0);
		free( (void*) strings[i]);
	}
	free(strings);
	return data;
}

void free_data(data_t* data, size_t num_objects){
	for(size_t i = 0; i < num_objects; i++){
		free_mat(&data[i].mat);
	}
	free(data);
}
