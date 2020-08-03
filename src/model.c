// keras style model functionality

#include "model.h"
#include "layer.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

model_t* create_model(){
	model_t* model = malloc(sizeof(model_t));
	//TODO: check if model is not NULL
	if(model == NULL){
		printf("ERROR: couldn't alloc memory for the model\n");
	}
	model->first_layer = NULL;
	return model;
}

model_t* sequential(int num_layers, int* num_nodes, Activation activation){
	model_t* model = create_model();

	if(num_layers < 2){
		printf("can't create model with less than two layers");
		return NULL;
	}	

	for(int i=1; i < num_layers; i ++){
		layer_t* layer = create_layer(num_nodes[i-1], num_nodes[i], activation);
		add_layer(model, layer);	
	}
	return model;	
}

void free_model(model_t** model){
	if(*model == NULL){
		printf("WARNING: trying to free a model that is allready freed!\n");
	}	
	// free all other layers.

	model_layer_t* c_layer = (*model)->first_layer;
	int found_last = 0;
	while(c_layer != NULL && !found_last){
		model_layer_t* layer_tmp = c_layer;
		found_last = c_layer->is_last;
		c_layer = c_layer->next_layer;
		free_layer(&layer_tmp->layer);
		free(layer_tmp);
	}
}

void add_layer(model_t* model, layer_t* layer){
	// check if the added layer is the first one. Then we create a new link list...
	model_layer_t* new_model_layer = malloc(sizeof(model_layer_t));
	if(new_model_layer == NULL){
		printf("ERROR: failed to alloc and therefore to add layer to model\n");
	}
	new_model_layer->is_last = 1;
	new_model_layer->layer = layer;
	if(model->first_layer == NULL){
		new_model_layer->next_layer = new_model_layer;
		new_model_layer->prev_layer = new_model_layer;
		model->first_layer = new_model_layer;
	} else {
		model_layer_t* last = get_last_layer(model);
		//Fix pointer
		last->is_last = 0;
		// connection to prev last layer
		last->next_layer = new_model_layer;
		new_model_layer->prev_layer = last;
		// connection to first layer
		new_model_layer->next_layer = model->first_layer;
		model->first_layer->prev_layer = new_model_layer;
	}
}

model_layer_t* get_last_layer(model_t* model){
	if(model->first_layer == NULL){
		return NULL;
	} else {
		return model->first_layer->prev_layer;
	}
}

tensor_handle_t* predict(model_t* model, tensor_handle_t* input){
	tensor_handle_t* next_input = input;
	model_layer_t* c_layer = model->first_layer;
	int found_last = 0;
	while(c_layer != NULL && !found_last){
		tensor_handle_t* tmp = forward_pass(c_layer->layer, next_input);
		// we free tensor that are in the middle
		if(input != next_input){
			free_tensor(&next_input);
		}
		next_input = tmp;
		found_last = c_layer->is_last;
		// go to next layer
		c_layer = c_layer->next_layer;
	}
	return next_input;
}

float* train(model_t* model, tensor_handle_t* input, tensor_handle_t* y, int epochs, float learning_rate){
	float* history = malloc(sizeof(float) * epochs);
	for(int epoch=0; epoch < epochs; epoch++){
		tensor_handle_t* model_output = predict(model, input);
		
		// print error
		tensor_handle_t* error = squared_loss(model_output, y);
		float sum = tensor_sum(error);
		int size = get_tensor_size(error);
		//printf("epoch %d loss: %f\n", epoch, sum/size);
		history[epoch] = sum/size;
		
		// TODO: support other error functions
		tensor_handle_t* next_error = squared_loss_derivative(model_output, y);	
		free_tensor(&model_output);
			
		// go backwards thru all the layers
		model_layer_t* c_layer = get_last_layer(model);
		int found_first = 0;
		while(c_layer != NULL && !found_first){
			tensor_handle_t* tmp = backward_pass(c_layer->layer, next_error, learning_rate);
			// we free the tensor that are in the middle
			free_tensor(&next_error);
			next_error = tmp;
			
			found_first = c_layer == model->first_layer;
			// go to prev layer
			c_layer = c_layer->prev_layer;
		}
		free_tensor(&next_error);
	}
	return history;
}

void save_weights(model_t* model, char* dir){
	struct stat st = {0};
	if (stat(dir, &st) == -1) {
		printf("creating dir \"%s\" for weights\n", dir);
		mkdir(dir, 0700);
	}
	char* filename = NULL;
	
	//TODO: handle allocs
	//TODO: also save the architecture of the model	
	model_layer_t* c_layer = model->first_layer;
	int layer_index = 0;
	int found_last = 0;	
	while(c_layer != NULL && !found_last){
		asprintf(&filename, "%s/%d.weights", dir, layer_index);
		printf("saving \"%s\"\n", filename);
		tensor_to_file(c_layer->layer->weights, filename);
		free(filename);
		asprintf(&filename, "%s/%d.bias", dir, layer_index);
		printf("saving \"%s\"\n", filename);
		tensor_to_file(c_layer->layer->bias, filename);
		free(filename);
		// go to next layer
		found_last = c_layer->is_last;
		c_layer = c_layer->next_layer;
		layer_index++;
	}
	printf("finished model saving.\n");
}

int load_weights(model_t* model, char* dir){
	char* filename = NULL;
	struct stat st = {0};
	if (stat(dir, &st) == -1) {
		printf("ERROR: trying to load model weights from dir \"%s\" that does not exist.\n", dir);
		return -1;
	}
	
	// iterate over all layers	
	model_layer_t* c_layer = model->first_layer;
	int layer_index = 0;
	int found_last = 0;	
	while(c_layer != NULL && !found_last){
		asprintf(&filename, "%s/%d.weights", dir, layer_index);
		printf("loading \"%s\"\n", filename);
		tensor_handle_t* weights = tensor_from_file(filename);
		//printf("loaded tensor:\n");
		//print_tensor(weights);
		// check if the size is equal.
		int size_new = get_tensor_size(weights);
		int size_old = get_tensor_size(c_layer->layer->weights);
		if(size_new == size_old){
			// we can replace the tensors and free the old one.
			// reshape the new tensor to the shape of the old one
			tensor_reshape(weights, c_layer->layer->weights->dims, c_layer->layer->weights->shape);
			free_tensor(&c_layer->layer->weights);
			c_layer->layer->weights = weights;	
		} else {
			printf("ERROR: failed to load weights for layer %d", layer_index);
			return -2;
		}
		free(filename);
		
		asprintf(&filename, "%s/%d.bias", dir, layer_index);
		printf("loading \"%s\"\n", filename);
		tensor_handle_t* bias = tensor_from_file(filename);
		// check if the size is equal.
		size_new = get_tensor_size(bias);
		size_old = get_tensor_size(c_layer->layer->bias);
		if(size_new == size_old){
			// we can replace the tensors and free the old one.
			// reshape the new tensor to the shape of the old one
			tensor_reshape(bias, c_layer->layer->bias->dims, c_layer->layer->bias->shape);
			free_tensor(&c_layer->layer->bias);
			c_layer->layer->bias = bias;	
		} else {
			printf("ERROR: failed to load bias for layer %d", layer_index);
			return -2;
		}
		free(filename);
		
		// go to next layer
		found_last = c_layer->is_last;
		c_layer = c_layer->next_layer;
		layer_index++;
	}
	printf("finished loading model.\n");
	return 0;
}

void print_model(model_t* model){
	printf("Model at %p\n", model);
	int layer_index = 0;
	
	model_layer_t* c_layer = model->first_layer;
	int found_last = 0;	
	while(c_layer != NULL && !found_last){
		int* shape = c_layer->layer->weights->shape;
		printf("Layer %d shape: [%d, %d]\n", layer_index, shape[0], shape[1]);
		// go to next layer
		found_last = c_layer->is_last;
		c_layer = c_layer->next_layer;
		layer_index++;
	}
}
