#include "csv_parsser.h"
#include "neural_net.h"

#define arr_size(arr) (sizeof((arr))/sizeof((arr[0])))

int main(int argc, char** argv){

	if(argc < 3){
		fprintf(stderr, "usage: %s <training csv file> <test csv file>\n", argv[0]);
		return EXIT_FAILURE;
	}

	const char* trainfilepath = argv[1];
	const char* testfilepath = argv[2];

	//TRAINING
	size_t number_imgs_trainedon, number_imgs_predicted;
	size_t num_layers[] = {INPUT_NODES, 16, 16, 10};
#if 1
	//printf("parssing %s\n", trainfilepath);
	data_t* imgs = csv_parss(trainfilepath, &number_imgs_trainedon);
	data_t* imgstobepredected = csv_parss(testfilepath, &number_imgs_predicted);
	//printf("kaynin %ld tswirra\n", number_imgs);

	neuralnet_t net;
	// Test different learning rates
	for(float i = 0.01f; i <= 0.5f; i += 0.01){
		double score = 0.l;
		for(uint32_t j = 0; j < 5; j++){
			init_net(&net, num_layers, arr_size(num_layers), i);

			// Train for multiple epochs
			//for(int epoch = 0; epoch < 5; epoch++)
				network_train_batch_imgs(&net, imgs, number_imgs_trainedon);

			// PREDICTING
			score += network_predict_imgs(&net, imgstobepredected, number_imgs_predicted);
			printf("%.2f, %.3f\r", i, (score / (j+1)));
			free_net(&net);
		}
		printf("\n");
	}
	free_data(imgs, number_imgs_trainedon);
	free_data(imgstobepredected, number_imgs_predicted);
	return 0;
#else
	data_t* imgs = csv_parss(trainfilepath, &number_imgs_trainedon);
	neuralnet_t net;

	init_net(&net, num_layers, arr_size(num_layers), 0.1);
	network_train_batch_imgs(&net, imgs, number_imgs_trainedon);

	// PREDICTING
	data_t* imgstobepredected = csv_parss(testfilepath, &number_imgs_predicted);
	double score = network_predict_imgs(&net, imgstobepredected, number_imgs_predicted);
	printf("Score: %1.5f\n", score);

	free_net(&net);
	free_data(imgs, number_imgs_trainedon);
	free_data(imgstobepredected, number_imgs_predicted);
	return 0;
#endif

}
