#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "tensor.h"

typedef struct model_layer{
	struct model_layer* next_layer;
	struct model_layer* prev_layer;
	layer_t* layer;
	int is_last;	
} model_layer_t;

typedef struct model {
	model_layer_t* first_layer;
	Activation activation;
} model_t;

model_t* create_model(Activation activation);
void free_model(model_t** model);
void add_layer(model_t* model, layer_t* layer);
model_layer_t* get_last_layer(model_t* model);
tensor_handle_t* predict(model_t* model, tensor_handle_t* input);
float* train(model_t* model, tensor_handle_t* input, tensor_handle_t* y, int epochs, float learning_rate);
void print_model(model_t* model);

#endif /* MODEL_H */
