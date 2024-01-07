#include "csv_parsser.h"
#include "neural_net.h"

#define arr_size(arr) (sizeof((arr))/sizeof((arr[0])))

int main(int argc, char** argv){

	//TRAINING
	size_t number_imgs;
	size_t num_layers[] = {784, 256, 10};
	
	struct Data* imgs = csv_parss("mnist_train.csv", &number_imgs);
	printf("kaynin %ld tswirra\n", number_imgs);
	struct NeuralNet net;
	
	init_net(&net, num_layers, arr_size(num_layers), 0.1);
	network_train_batch_imgs(&net, imgs, number_imgs);

	// PREDICTING
	struct Data* imgstobepredected = csv_parss("mnist_test.csv", &number_imgs);
	double score = network_predict_imgs(&net, imgstobepredected, number_imgs);
	printf("Score: %1.5f\n", score);

	free_net(&net);
	return 0;

}

