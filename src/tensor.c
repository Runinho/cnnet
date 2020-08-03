#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "layer.h"

/*int main(){
	int shape[3];
	shape[0] = 4;
	shape[1] = 3;
	shape[2] = 2;

	tensor_handle_t* t = create_tensor(2, shape);
	print_tensor(t);
	tensor_add(t, 5);
	print_tensor(t);
	free_tensor(&t);
	
	tensor_handle_t* a = tensor_arange(0, 60, 1);
	int new_shape[] = {3, 4, 5};
	print_tensor(a);
	tensor_reshape(a, 3, new_shape);
	print_tensor(a);
	
	int position[] = {2,3,3};
	printf("get element %f \n", tensor_get(a, position));

	free_tensor(&a);
	
	a = tensor_arange(0,12,1);
	tensor_handle_t* b = tensor_arange(0,12,1);
	int three_by_three[] = {3, 4};
	tensor_reshape(a, 2, three_by_three);
	
	three_by_three[0] = 4;
	three_by_three[1] = 3;
	tensor_reshape(b, 2, three_by_three);
	
	tensor_handle_t* c = tensor_mat_multiply(a, b);	

	print_tensor(a);
	print_tensor(b);
	print_tensor(c);
	free_tensor(&a);
	free_tensor(&b);
	free_tensor(&c);

	a = tensor_arange(-5,5,0.33);
	print_tensor(a);
	relu(a);
	print_tensor(a);
	free_tensor(&a);	

	three_by_three[0] = 10;
	three_by_three[1] = 5;
	a = tensor_random_range(2, three_by_three, -1, 1);
	print_tensor(a);	
	
	tensor_add(a, 0);	
	b = tensor_elm_multiply(a, a);
	print_tensor(b);	

	float sum = tensor_sum(a);
	printf("tensor sum: %f", sum);
	int size = get_tensor_size(b);
	printf("tensor avg: %f", sum/((float)size));
	free_tensor(&a);
	free_tensor(&b);
		
	a = tensor_arange(-5, 5, 0.25);
	print_tensor(a);
	tensor_sigmoid(a);
	print_tensor(a);
	free_tensor(&a);

	a = tensor_arange(-5, 5, 0.25);
	print_tensor(a);
	tensor_sigmoid_derivative(a);
	print_tensor(a);
	free_tensor(&a);

	// test elm_add
	a = tensor_random(3, (int[]){10, 2, 4});
	print_tensor(a);

	b = tensor_arange(0, 4, 1);	
	tensor_reshape(b, 3, (int[]){1, 1, 4});
	print_tensor(b);
	tensor_elm_add(a, b);
	print_tensor(a);	

	free_tensor(&b);
	free_tensor(&a);

	printf("Hallo World %p", NULL);
}*/

// memory managment
tensor_handle_t* create_tensor(int dims, int*shape){
	if(dims <= 0){
		printf("ERROR in Tensor: can't create tensor with negative dimensions");
		return NULL;
	}
	int data_size = 1;

	// check if all dimensions are creater than 1
	for(int i=0; i < dims; i++){
		if(shape[i] <= 0){
			printf("ERROR in Tensor: can't create tensor with a shape that is not creater than zero");
		}
		data_size *= shape[i];
	}
	
	// alloc space for our handle
	tensor_handle_t* handle = malloc(sizeof(tensor_handle_t));
	if(handle == NULL){
		return NULL;
	}
	
	handle->dims = dims;
	
	// alloc for shape
	int* shape_cpy = malloc(sizeof(int) * dims);
	if(shape_cpy == NULL){
		free(handle);
		return NULL;
	}
	// copy shape
	for(int i=0; i < dims; i++){
		shape_cpy[i] = shape[i];
	}
	handle->shape = shape_cpy;
	
	// alloc for stride
	int* stride_cpy = malloc(sizeof(int) * dims);
	if(stride_cpy == NULL){
		free(handle);
		free(shape_cpy);
		return NULL;
	}
	handle->stride = stride_cpy;
	stride_from_shape(stride_cpy, dims, shape_cpy);
		
	// alloc for data
	float* data = malloc(sizeof(float) * data_size);
	if(data == NULL){
		free(handle);
		free(shape_cpy);
		free(stride_cpy);
		return NULL;
	}

	handle->data = data;	
	
	//printf("created Tensor at %p\n", handle);
		
	return handle;
}

tensor_handle_t* tensor_from_file(char* filename){
	// reads the raw float32 data from the filename and returns a flat array.
	FILE *file;
	file = fopen(filename, "rb");
	fseek(file, 0, SEEK_END); // seek to end of file
	int file_size = ftell(file); // get current file pointer
	fseek(file, 0, SEEK_SET); // seek to the begining
	int elements = file_size / sizeof(float);
	if(file_size != elements * sizeof(float)){
		printf("WARNING: File \"%s\" size(%d) is not even devisable by %lu. Is this raw float32 data?", filename, file_size, sizeof(float));
	}
	tensor_handle_t* handle = create_tensor(1, (int[]){elements});	
	fread(handle->data, sizeof(float), elements, file);
	fclose(file);
	return handle;	
}

void tensor_to_file(tensor_handle_t* handle, char* filename){
	FILE *file;
	file = fopen(filename, "wb");
	int size = get_tensor_size(handle);
	fwrite(handle->data, sizeof(float), size, file);
	printf("wrote tensor to %s\n", filename);
	fclose(file);
}

void free_tensor(tensor_handle_t** handle){
	if(*handle == NULL){
		printf("WARNING: trying to free NULL handle\n");
		return;
	}
	free((*handle)->data);
	free((*handle)->shape);
	free((*handle)->stride);
	free(*handle);
	*handle = NULL;
}

int calc_tensor_size(int dims, int* shape){
	int size = 1;
	for(int i=0; i < dims;i++){
		size *= shape[i];
	}
	return size;
}

int get_tensor_size(tensor_handle_t* handle){
	return calc_tensor_size(handle->dims, handle->shape);
}

int tensor_reshape(tensor_handle_t* handle, int dims, int* new_shape){
	int old_size = get_tensor_size(handle);
	int new_size = calc_tensor_size(dims, new_shape);
	if(new_size != old_size){
		printf("can't reshape array with size %d to an array with size %d \n", old_size, new_size);
		printf("old ");
		print_tensor_shape(handle);
		printf("new ");
		print_int_array(dims, new_shape);
		return -1;
	}
	// TODO: add check if new dimension is positiv on all positions	
	int* shape_cpy = malloc(sizeof(int) * dims);
	if(shape_cpy == NULL){
		printf("can't reshape because new dimension couldn't be allocated.");
		// TODO: handle this error better
	}
	for(int i=0; i < dims; i++){
		shape_cpy[i] = new_shape[i];
	}	
	
	handle->dims = dims;
	free(handle->shape);
	handle->shape = shape_cpy;

	stride_from_shape(handle->stride, dims, handle->shape);
	return 1;
}

void flip_array(int length, int* array){
	int half_dims = length/2;
	for(int i=0; i < half_dims; i++){
		int tmp = array[i];
		array[i] = array[length-i-1];
		array[length-i-1] = tmp;
	}
}

void tensor_transpose(tensor_handle_t* handle){
	// we just have to "flip" the stride and shape;
	flip_array(handle->dims, handle->stride);
	flip_array(handle->dims, handle->shape);
}

tensor_handle_t* tensor_copy(tensor_handle_t* handle){
	tensor_handle_t* res = create_tensor(handle->dims, handle->shape);
	if(res == NULL){
		printf("ERROR: failed to allocate memory in copy");
		return NULL;
	}
	// copy stride
	int* new_stride = malloc(sizeof(int) * handle->dims);
	if(new_stride == NULL){
		printf("ERROR: failed to allocate memory in copy");
		free_tensor(&res);
		return NULL;
	}

	free(res->stride);
	for(int i=0; i < handle->dims; i++){
		new_stride[i] = handle->stride[i];
	}
	res->stride = new_stride;	

	//copy data
	int size = get_tensor_size(res);
	for(int i=0; i < size; i++){
		res->data[i] = handle->data[i];	
	}	
	return res;
}

int tensor_equal(tensor_handle_t* a, tensor_handle_t* b){
	if(a->dims != b->dims){
		return 0;
	}
	// check if stride and shape is equal
	for(int i=0; i < a->dims; i++){
		if(a->shape[i] != b->shape[i] || a->stride[i] != b->stride[i]){
			return 0;
		}
	}
	
	// check data
	int size = get_tensor_size(a);
	for(int i=0; i < size; i++){
		if(a->data[i] != b->data[i]){
			return 0;
		}
	}
	return 1;
}

// numpy style creation

tensor_handle_t* tensor_arange(float start, float stop, float step){
	// calc size:
	int size = (int)((stop - start)/step);
	if(size <= 0){
		printf("ERROR: can't from %f to %f with step %f", start, stop, step);
		return NULL;
	}
	
	tensor_handle_t* res = create_tensor(1, &size);

	float value = start;
	for(int i=0; i < size; i++){
		res->data[i] = value;
		value += step;
	}
	return res; 
}

tensor_handle_t* tensor_random_range(int dims, int* shape, float start, float stop){
	tensor_handle_t* res = create_tensor(dims, shape);
	int size = get_tensor_size(res);
	float divide = 1.0/2147483647; //devided by (2**31)-1
	divide *= (stop-start);
	//printf("divide by: %f\n", divide);
	for(int i=0; i < size; i++){
		res->data[i] = ((float)random()) * divide + start;
	}
	return res;
}

tensor_handle_t* tensor_random(int dims, int* shape){
	return tensor_random_range(dims, shape, 0, 1);
}

// indexing
void stride_from_shape(int* stride, int dims, int* shape){
	int size = 1;
	for(int dim_i=dims-1; dim_i >= 0; dim_i--){
		stride[dim_i] = size;
		size *= shape[dim_i];
	}
}

int tensor_get_index(tensor_handle_t* handle, int* indices){
	// TODO: do indexing with pre calculated strides
	int index = 0;
	int size = 1;
	for(int dim_i=0; dim_i < handle->dims; dim_i++){
		index += handle->stride[dim_i] * indices[dim_i];
	}
	return index;
}

float tensor_get(tensor_handle_t* handle, int* indices){
	int index = tensor_get_index(handle, indices); 
	return handle->data[index];
}

float* tensor_get_p(tensor_handle_t* handle, int* indices){
	int index = tensor_get_index(handle, indices); 
	return &(handle->data[index]);
}

// math
void tensor_add(tensor_handle_t* handle, float to_add){
	int tensor_size = get_tensor_size(handle);
	for(int i=0; i < tensor_size; i++){
		handle->data[i] += to_add;
	}
}

void tensor_elm_add(tensor_handle_t* a, tensor_handle_t* b){
	// add values from b to a. with broadcasting.
	// element wise add, but if dimensions in b are one just always use that.
	// check if arrays dims are equal.
	if(a->dims != b->dims){
		printf("ERROR: can't elmentwise add two arrays with dimensions %d and %d", a->dims, b->dims);
		//TODO: exit programm
		return;
	}
	// check if broadcastable
	for(int dims_i=0; dims_i < a->dims; dims_i++){
		if(a->shape[dims_i] != b->shape[dims_i] && b->shape[dims_i] != 1){
			printf("ERROR: can't broadcast shapes in add.");
			print_tensor_shape(a);
			print_tensor_shape(b);
		}
	}
	
	// we iterate of the the elements of a
	int* position_a = malloc(sizeof(int) * a->dims);
	int* position_b = malloc(sizeof(int) * a->dims);
	int size = get_tensor_size(a);
	for(int i=0; i < size; i++){
		// calculate the position (indices)
		int remainder = i;
		for(int dim_i=0; dim_i < a->dims; dim_i++){
			position_a[dim_i] = remainder/(a->stride[dim_i]);
			remainder = remainder % (a->stride[dim_i]);
			//printf("i: %d, remainder %d, dim_i: %d positions[dim_i]: %d \n", i, remainder, dim_i, position_a[dim_i]);
			if(b->shape[dim_i] != 1){
				position_b[dim_i] = position_a[dim_i];
			}
			else {
				position_b[dim_i] = 0;
			}
		}
		float to_add = tensor_get(b, position_b);
		a->data[i] += to_add;	
	}
	free(position_a); 
	free(position_b); 
}

void tensor_multiply(tensor_handle_t* handle, float scalar){
	int tensor_size = get_tensor_size(handle);
	for(int i=0; i < tensor_size; i++){
		handle->data[i] *= scalar;
	}
}

tensor_handle_t* tensor_mat_multiply(tensor_handle_t* a, tensor_handle_t* b){
	// calculate resulting shape
	if(a->dims != b-> dims){
		printf("ERROR: Multiply not implemented if dim from a != dim from b");
		return NULL;
	}
	if(a->dims !=2 || b->dims != 2){
		printf("ERROR: Multiply not implemented if dimension is not 2");
		return NULL;
	}
	if(a->shape[1] != b->shape[0]){
		printf("ERROR: Tensor shapes do not match for multiplication");
		return NULL;
	}
	
	int new_shape[2];
	new_shape[0] = a->shape[0];
	new_shape[1] = b->shape[1];
	
	tensor_handle_t* res = create_tensor(2, new_shape);
	// create new handle

	int data_index = 0;
	for(int i=0; i < new_shape[0]; i++){
		for(int j=0; j < new_shape[1]; j++){
			//TODO: do the multiplication..
			float sum = 0;
			int position_a[2];
			int position_b[2];
			
			position_a[0] = i;
			position_b[1] = j;
			#pragma clang loop vectorize(enable)
			for(int sum_i=0; sum_i < a->shape[1]; sum_i++){
				position_a[1] = sum_i;
				position_b[0] = sum_i;
				float elm_a = tensor_get(a, position_a);
				float elm_b = tensor_get(b, position_b);
				//printf("i:%d j:%d elm_a:%f elm_b:%f sum:%f\n\n", i, j, elm_a, elm_b, sum);
				sum += elm_a * elm_b;
			}
			//printf("i:%d j:%d sum:%f\n\n", i, j, sum);
			res->data[data_index] = sum;
			data_index++;
		}
	}
	return res;
}

float tensor_sum(tensor_handle_t* handle){
	float sum = 0;
	int size = get_tensor_size(handle);
	
	#pragma clang loop vectorize(enable)
	for(int i=0; i < size; i++){
		sum += handle->data[i];
	}
	return sum;
}

tensor_handle_t* tensor_elm_multiply(tensor_handle_t* a, tensor_handle_t* b){
	if(a->dims != b->dims){
		printf("ERROR: Multiply not implemented if dim from a != dim from b");
		return NULL;
	}
	// check if dimensions and strides are equal
	// TODO: create method that does this operation if stride is not equal.
	for(int dims_i=0; dims_i < a->dims; dims_i++){
		if(a->shape[dims_i] != b->shape[dims_i]){
			printf("ERROR: Multiply not implemented if shape from a != shape from b");
			return NULL;
		}
		if(a->stride[dims_i] != b->stride[dims_i]){
			printf("ERROR: Multiply not implemented if stride from a != stride from b");
			return NULL;
		}
	}
	
	tensor_handle_t* res = create_tensor(a->dims, a->shape);
	int size = get_tensor_size(res);
	for(int i=0; i < size; i++){
		res->data[i] = a->data[i] * b->data[i];
	}	
	return res;
}	

//activation functions
void tensor_sigmoid(tensor_handle_t* handle){
	// we use the following sigmoid function: f(x)= x/(1 + abs(x))
	int size = get_tensor_size(handle);
	for(int i=0; i < size; i++){
		float x = handle->data[i];
		// do the computation
		x = x/(1 + fabsf(x));
		handle->data[i] = x;
		
	} 
}

void tensor_sigmoid_derivative(tensor_handle_t* handle){
	// we use the following sigmoid function: f(x)= x/(1 + abs(x))
	// derivative is: (1 + abs(x) - abs'(x) * x))/((1 + abs(x))**2)
	int size = get_tensor_size(handle);
	for(int i=0; i < size; i++){
		float x = handle->data[i];
		// do the computation
		float v = 1 + fabsf(x);
		float v_tick = (x < 0 ? -1 : 1); // not 100% correct abs(0) = 0....
		x = (v - v_tick * x)/(v * v);
		handle->data[i] = x;
	}
}

void tensor_relu(tensor_handle_t* a){
	int size = get_tensor_size(a);
	for(int i=0; i < size; i++){
		if(a->data[i]<=0){
			a->data[i] = 0;
		}
	}
}


void tensor_relu_derivative(tensor_handle_t* a){
	int size = get_tensor_size(a);
	for(int i=0; i < size; i++){
		if(a->data[i]<=0){
			a->data[i] = 0;
		} else {
			a->data[i] = 1;
		}
	}
}

// prints
void print_int_array(int length, int* a){
	printf(": [");
	for(int i=0; i < length; i++){
		printf("%d", a[i]);
		if(i < length - 1){
			printf(", ");
		}
	}
	printf("]\n");
}

void print_tensor_stride(tensor_handle_t* handle){
	printf("stride");
	print_int_array(handle->dims, handle->stride);
}


void print_tensor_shape(tensor_handle_t* handle){
	printf("shape");
	print_int_array(handle->dims, handle->shape);
}

void print_tensor(tensor_handle_t* handle){
	printf("Tensor at %p: \n", handle);
	if(handle == NULL){
		printf("Tensor handle is null\n");
		return;
	}
	print_tensor_shape(handle);
	print_tensor_stride(handle);
	printf("data: \n");
	int* position = malloc(sizeof(int) * handle->dims);
	if(position == NULL){
		printf("failed to malloc position array. canceling print");
		return;
	}
	int* default_stride = malloc(sizeof(int) * handle->dims);
	if(default_stride == NULL){
		printf("failed to malloc position array. canceling print");
		return;
	}
	stride_from_shape(default_stride, handle->dims, handle->shape);	
	int tensor_size = get_tensor_size(handle);
	int shifts = 0;
	for(int i=0; i < tensor_size; i++){
		// check if dims end...
		int moduler = 1;
		int closer = 0;
		for(int dims_i=handle->dims -1; dims_i >= 0;dims_i--){
			moduler *= handle->shape[dims_i]; // this should be the default stride
			int mod = i % moduler;
			if(mod == 0){
				printf("[");
				shifts++;
			}
			if(mod == moduler - 1){
				closer++;
			}
		} 
		// calculate index with respect to stride
		//TODO: check if stride is the default one... and then set index to i	
		// we can just calculate the position with the data above
		int remainder = i;
		for(int dim_i=0; dim_i < handle->dims; dim_i++){
			position[dim_i] = remainder/(default_stride[dim_i]);
			remainder = remainder % (default_stride[dim_i]);
		}
		int index = tensor_get_index(handle, position);	
		printf("%f", handle->data[index]);
		for(int j=0; j < closer; j++){
			printf("]");
			shifts--;
		}
		if(closer == 0){
			printf(", ");
		} else {
			printf("\n");
			for(int shift_i=0; shift_i<shifts; shift_i++){
				printf(" ");
			}
		}
	} 	
	printf("\n");
	free(position);
}
