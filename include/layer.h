#include "tensor.h"

#ifndef LAYER_H
#define LAYER_H

// activation function
typedef void (*activationFunc)(tensor_handle_t*);

typedef enum activation {
	relu,
	sigmoid,
} Activation;


typedef struct layer{
	tensor_handle_t* weights;
	tensor_handle_t* bias;
	tensor_handle_t* last_pre_sigmoid;
	tensor_handle_t* last_input;
	Activation activation;
} layer_t;

tensor_handle_t* squared_loss(tensor_handle_t* output, tensor_handle_t* y);
tensor_handle_t* squared_loss_derivative(tensor_handle_t* output, tensor_handle_t* y);

activationFunc get_activation(Activation activation);
activationFunc get_activation_derivative(Activation activation);

layer_t* create_layer(int num_inputs, int outputs, Activation activation);
void free_layer(layer_t** handle);
tensor_handle_t* forward_pass(layer_t* layer, tensor_handle_t* input);
tensor_handle_t* backward_pass(layer_t* layer, tensor_handle_t* error, float learning_rate);

#endif /* LAYER_H */
