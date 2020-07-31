#include "layer.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

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

activationFunc get_activation(Activation activation){
	switch(activation){
		case relu:
			return tensor_relu;
		case sigmoid:
			return tensor_sigmoid;
	}
}

activationFunc get_activation_derivative(Activation activation){
	switch(activation){
		case relu:
			return tensor_relu_derivative;
		case sigmoid:
			return tensor_sigmoid_derivative;
	}
}

layer_t* create_layer(int num_inputs, int outputs, Activation activation){
	//TODO: do random intialization
	//create new layer:
	layer_t* handle = malloc(sizeof(layer_t));
	if(handle == NULL){
		printf("couldn't alloc layer");
		return NULL;
	}
	int weights_shape[2] = {num_inputs, outputs};
	
	// TODO: research weights initialization
	float start_weights = -0.5;
	float stop_weights = 0.5;
	if(activation == relu){
		start_weights = 0;
		stop_weights = 1;
	}
	handle->weights = tensor_random_range(2, weights_shape, start_weights, stop_weights);

	int bias_shape[2] = {1, outputs};
	handle->bias = tensor_random_range(2, bias_shape, start_weights, stop_weights);	
	handle->last_pre_sigmoid = NULL;
	handle->last_input = NULL;
	handle->activation = activation;
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
	//tensor_sigmoid(res);
	(*get_activation(layer->activation))(res);
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
	(*get_activation_derivative(layer->activation))(layer->last_pre_sigmoid);
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
