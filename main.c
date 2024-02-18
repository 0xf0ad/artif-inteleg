#include "csv_parsser.h"
#include "neural_net.h"

#define arr_size(arr) (sizeof((arr))/sizeof((arr[0])))

int main(int argc, char** argv){

	if(argc < 3){
		fprintf(stderr, "usage: %s <traning csv file> <test csv file>\n", argv[0]);
		return EXIT_FAILURE;
	}

	const char* trainfilepath = argv[1];
	const char* testfilepath = argv[2];

	//TRAINING
	size_t number_imgs;
	size_t num_layers[] = {784, 256, 10};
	
	printf("parssing %s\n", trainfilepath);
	data_t* imgs = csv_parss(trainfilepath, &number_imgs);
	printf("kaynin %ld tswirra\n", number_imgs);
	neuralnet_t net;
	
	init_net(&net, num_layers, arr_size(num_layers), 3.f);
	network_train_batch_imgs(&net, imgs, number_imgs);

	// PREDICTING
	printf("parssing %s\n", testfilepath);
	data_t* imgstobepredected = csv_parss(testfilepath, &number_imgs);
	double score = network_predict_imgs(&net, imgstobepredected, number_imgs);
	printf("Score: %1.5f%% accuracy\n", score * 100.f);

	free_net(&net);
	return 0;

}
