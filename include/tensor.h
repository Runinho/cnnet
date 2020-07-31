#ifndef TENSOR_H
#define TENSOR_H

// TODO: put the ifdef stuff here

typedef struct tensor_handle{
	float* data;
	int dims;
	int* shape;
	int* stride;
} tensor_handle_t;

tensor_handle_t* create_tensor(int dims, int*shape);
void free_tensor(tensor_handle_t** handle); 
int tensor_reshape(tensor_handle_t* handle, int dims, int* new_shape);
tensor_handle_t* tensor_arange(float start, float stop, float step);
tensor_handle_t* tensor_random(int dims, int* shape);
tensor_handle_t* tensor_random_range(int dims, int* shape, float start, float stop);
tensor_handle_t* tensor_from_file(char* filename);
void tensor_to_file(tensor_handle_t* handle, char* filename);
tensor_handle_t* tensor_copy(tensor_handle_t* handle);
int tensor_equal(tensor_handle_t* a, tensor_handle_t* b);
int get_tensor_size(tensor_handle_t* handle);

void stride_from_shape(int* stride, int dims, int* shape);
int tensor_get_index(tensor_handle_t* handle, int* indices);
float tensor_get(tensor_handle_t* handle, int* indices);
float* tensor_get_p(tensor_handle_t* handle, int* indices);

void print_tensor(tensor_handle_t* handle);
void print_tensor_shape(tensor_handle_t* handle);
void print_tensor_stride(tensor_handle_t* handle);
void print_int_array(int length, int* a);

void tensor_transpose(tensor_handle_t* handle);
void tensor_add(tensor_handle_t* handle, float to_add);
void tensor_elm_add(tensor_handle_t* a, tensor_handle_t* b);
void tensor_multiply(tensor_handle_t* handle, float scalar);
tensor_handle_t* tensor_mat_multiply(tensor_handle_t* a, tensor_handle_t* b);
tensor_handle_t* tensor_elm_multiply(tensor_handle_t* a, tensor_handle_t* b);
float tensor_sum(tensor_handle_t* handle);
void tensor_sigmoid(tensor_handle_t* handle);
void tensor_sigmoid_derivative(tensor_handle_t* handle);

#endif /* TENSOR_H */
