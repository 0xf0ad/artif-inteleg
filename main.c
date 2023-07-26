#include "csv_parsser.h"
#include "neural_net.h"

#define arr_size(arr) (sizeof((arr))/sizeof((arr[0])))

int main(void){

	//csv_parss("mnist_test.csv");

	//return 0;

	//TRAINING
	size_t number_imgs = 2000;
	size_t num_layers[] = {784, 256, 10};
	
	struct Data* imgs = csv_parss("mnist_test.csv", &number_imgs);
	printf("kaynin %d tswirra\n", number_imgs);
	struct NeuralNet net;
	
	init_net(&net, num_layers, arr_size(num_layers), 0.1);
	network_train_batch_imgs(&net, imgs, 1000);

	// PREDICTING
	number_imgs = 3000;
	double score = network_predict_imgs(&net, imgs, number_imgs);
	printf("Score: %1.5f\n", score);

	//(imgs, number_imgs);
	//network_free(net);
	return 0;

}

