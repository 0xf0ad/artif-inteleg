#ifndef CSV_PARSS_H_
#define CSV_PARSS_H_

#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// the number of nodes in the input layer
// it is fixed because I will be trainning only the MNIST set
// witch compose of 28x28 pixels = 628 pixels
#define IMG_SIZE     28
#define INPUT_NODES  IMG_SIZE * IMG_SIZE // 784

struct Data{
	uint8_t label;
	struct Matrix mat;
};

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
		fprintf(stderr, "ERROR: bro buy your new potatos\n");
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

struct Data* csv_parss(const char* filename, size_t* p_num_objects){
	size_t num_objects;
	const char** strings = extract_lines(filename, &num_objects);
	*p_num_objects = num_objects;

	struct Data* data = (struct Data*) malloc(sizeof(struct Data) * num_objects);
	for(size_t i = 0; i != num_objects; i++){
		// we assume the label is always a single digit so one character
		// so we convert a single character
		data[i].label = strings[i][0] - '0';
		// we expect consumming a char for the label and an other for the comma ,
		const char* string_offset = (strings[i] + 2);
		init_mat(&data[i].mat, INPUT_NODES + 1, 1);
		for(size_t j = 0; j != INPUT_NODES; j++){
			sscanf(string_offset, "%lf", &data[i].mat.entries[j]);
			// we add a one to encounter the comma character
			string_offset += strcspn(string_offset, ",") + 1;
		}
		// that is for bias
		// I know it should not be on an input layer but being there just simplify thinks
		data[i].mat.entries[INPUT_NODES] = 1.0l;
		// to devide every element by 255 to get to the range 0 to 1
		dev_mat_scalar(&data[i].mat, 255);
		//printf("LABEL : %d\n", data[i].label);
		//print_mat(&data[i].mat);
		free( (void*) strings[i]);
	}
	free(strings);
	return data;
}

// dont use this it is from chat-GPT
// AI is widely known for its unsecurity
// I m talking like my function cointain zero vulnrabilities
// if my elf file was provided in a CTF it will be cracked in 3 seconds
char** read_file_lines(const char* filename, int* line_count) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		printf("Failed to open the file.\n");
		return NULL;
	}

	char** lines = NULL;
	char buffer[256];
	int count = 0;

	// Count the number of lines in the file
	while (fgets(buffer, sizeof(buffer), file) != NULL) {
		count++;
	}

	// Allocate memory for the lines array
	lines = (char**)malloc(count * sizeof(char*));
	if (lines == NULL) {
		printf("Failed to allocate memory.\n");
		fclose(file);
		// cleanning the memory will be a sufisticated process
		// instead I wioll terminate the program the the memory will be reclaimed by the OS
		// assuming you dont use windws
		// dont use it
		return NULL;
	}

	// Reset the file position indicator to the beginning of the file
	rewind(file);

	// Read each line from the file and store it in the lines array
	for (int i = 0; i < count; i++) {
		fgets(buffer, sizeof(buffer), file);
		buffer[strcspn(buffer, "\n")] = '\0'; // Remove trailing newline character

		// Allocate memory for the line
		lines[i] = (char*)malloc((strlen(buffer) + 1) * sizeof(char));
		if (lines[i] == NULL) {
			printf("Failed to allocate memory.\n");
			fclose(file);
			for (int j = 0; j < i; j++) {
				free(lines[j]);
			}
			free(lines);
			return NULL;
		}

		// Copy the line to the lines array
		strcpy(lines[i], buffer);
	}

	*line_count = count;
	fclose(file);
	return lines;
}




#endif
