#include "layer.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

// currently not used
void relu(tensor_handle_t* a){
	int size = get_tensor_size(a);
	for(int i=0; i < size; i++){
		if(a->data[i]<=0){
			a->data[i] = 0;
		}
	}
}


void relu_derivative(tensor_handle_t* a){
	int size = get_tensor_size(a);
	for(int i=0; i < size; i++){
		if(a->data[i]<=0){
			a->data[i] = 0;
		} else {
			a->data[i] = 1;
		}
	}
}

tensor_handle_t* squared_loss(tensor_handle_t* output, tensor_handle_t* y){
	tensor_handle_t* error = tensor_copy(y);
	tensor_multiply(error, -1);
	tensor_elm_add(error, output);
	tensor_handle_t* ret = tensor_elm_multiply(error, error);
	// TODO: do the multiplication inplace 
	free_tensor(&error);
	return ret;
}

tensor_handle_t* squared_loss_derivative(tensor_handle_t* output, tensor_handle_t* y){
	tensor_handle_t* error = tensor_copy(y);
	tensor_multiply(error, -1);
	tensor_elm_add(error, output);
	return error;
}

layer_t* create_layer(int num_inputs, int outputs){
	//TODO: do random intialization
	//create new layer:
	layer_t* handle = malloc(sizeof(layer_t));
	if(handle == NULL){
		printf("couldn't alloc layer");
		return NULL;
	}
	int weights_shape[2] = {num_inputs, outputs};
	handle->weights = tensor_random_range(2, weights_shape, -0.5, 0.5);

	int bias_shape[2] = {1, outputs};
	handle->bias = tensor_random_range(2, bias_shape, -0.5, 0.5);	
	handle->last_pre_sigmoid = NULL;
	handle->last_input = NULL;
	return handle;
}

void free_layer(layer_t** handle){
	if(*handle == NULL){
		printf("WARNING: freeing NULL layer");
	}
	free((*handle)->weights);
	free((*handle)->bias);
	free((*handle)->last_pre_sigmoid);
	free((*handle)->last_input);
	free(*handle);
	*handle = NULL;
}

tensor_handle_t* forward_pass(layer_t* layer, tensor_handle_t* input){
	// copy for backpropagation
	if(layer->last_input != NULL){
		free_tensor(&layer->last_input);
	}
	layer->last_input = tensor_copy(input);
	// the returned array is also used in the backward pass
	tensor_handle_t* res = tensor_mat_multiply(input, layer->weights);	
	// copy for back propagation
	tensor_elm_add(res, layer->bias);
	if(layer->last_pre_sigmoid != NULL){
		free_tensor(&layer->last_pre_sigmoid);
	}
	layer->last_pre_sigmoid = tensor_copy(res);
	
	// activation function
	// TODO: implement other activation functions
	//relu(res);
	tensor_sigmoid(res);
	return res;
}

tensor_handle_t* backward_pass(layer_t* layer, tensor_handle_t* error, float learning_rate){
	// we modifiy layer_pre_sigmoid. 
	if(layer->last_input == NULL || layer->last_pre_sigmoid == NULL){
		printf("can't do backprop because the one or more requiered tensor were freed.");
	}
	// Shapes:
	// error: [batch, outputs]
	// last_pre_sigmoid: [batch_outputs]
	// delta: [batch, outputs]
	// weights: [inputs, outputs]
	// next_error: delta x weights.T
	
	// calculate grad: error * sigmoid'(layer_pre_sigmoid)
	// apply derivative of sigmoid
	//relu_derivative(layer->last_pre_sigmoid);
	tensor_sigmoid_derivative(layer->last_pre_sigmoid);
	tensor_handle_t* delta = tensor_elm_multiply(layer->last_pre_sigmoid, error);
	
	// calculate next error: error x weights.T
	tensor_transpose(layer->weights);
	tensor_handle_t* next_error = tensor_mat_multiply(error, layer->weights);
	tensor_transpose(layer->weights);
	
	//update weights
	tensor_transpose(layer->last_input);
	tensor_handle_t* grad = tensor_mat_multiply(layer->last_input, delta);
	tensor_transpose(layer->last_input);
	// create updates
	tensor_multiply(grad, -1 * learning_rate);
	tensor_elm_add(layer->weights, grad);
	
	// for bias: ones with shape[1, num_b] x delta;
	tensor_handle_t one_tensor;
	int shape[2] = {1, delta->shape[0]};
	int stride[2] = {0, 0};	
	float one = 1;
	one_tensor.dims = 2;
	one_tensor.shape = (int*)&shape;
	one_tensor.stride = (int*)&stride;
	one_tensor.data = &one;

	tensor_handle_t* grad_bias = tensor_mat_multiply(&one_tensor, delta);
	tensor_multiply(grad_bias, -1 * learning_rate);
	tensor_elm_add(layer->bias, grad_bias);
	
	
	free_tensor(&delta);
	free_tensor(&grad);
	free_tensor(&grad_bias);
	return next_error;
}
