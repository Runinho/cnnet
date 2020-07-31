// small tests for the framwork
#include "tensor.h"
#include "layer.h"
#include <stdio.h>

//typedef int (*test_f)();

int tests = 0;
int failed = 0;

void expect_f(float actual, float expect, char* s){
	tests++;
	if(actual < expect - 0.00001 ||actual > expect + 0.00001){
		printf("❗️TEST FAILED. %s: expected %f to be %f\n", s, actual, expect);
		failed++;
	} else {
		printf("✅ %s\n", s);
	}
}

void expect(int actual, int expect, char* s){
	tests++;
	if(actual != expect){
		printf("❗️TEST FAILED. %s: expected %d to be %d\n", s, actual, expect);
		failed++;
	} else {
		printf("✅ %s\n", s);
	}
}

void test_equal(){
	tensor_handle_t* a = tensor_arange(0, 10, 1);
	tensor_handle_t* b = tensor_arange(0, 10, 1);
	expect(tensor_equal(a, b), 1, "equal test same");
	
	a->data[3] = 47; 
	expect(tensor_equal(a, b), 0, "equal test data not same");
	
	free_tensor(&a);
	free_tensor(&b);
	expect((int)a, (int)NULL, "check for NULL");
}

void test_creation(){
	tensor_handle_t* a = create_tensor(3, (int[]){3, 4, 5});
	
	expect(a->dims, 3, "creation dims");
	expect(a->shape[0], 3, "creation shape 0");
	expect(a->shape[1], 4, "creation shape 1");
	expect(a->shape[2], 5, "creation shape 2");

	expect(a->stride[0], 4*5, "creation stride 0");
	expect(a->stride[1], 5, "creation stride 1");
	expect(a->stride[2], 1, "creation stride 2");
	free_tensor(&a);
}

void test_add(){
	tensor_handle_t* a = tensor_arange(0, 3, 1);
	tensor_handle_t* b = tensor_arange(3, 6, 1);
	tensor_add(a, 3);
	expect(tensor_equal(a, b), 1, "tensor add");
	free_tensor(&a);
	free_tensor(&b);
}

void test_scalar_multiply(){
	tensor_handle_t* a = tensor_arange(0, 3, 1);
	tensor_handle_t* b = tensor_arange(0, 6, 2);
	tensor_multiply(a, 2);
	expect(tensor_equal(a, b), 1, "tensor scalar multiply");
	free_tensor(&a);
	free_tensor(&b);
}

void test_get(){
	tensor_handle_t* a = tensor_arange(0, 3*4*5, 1);
	expect(tensor_get(a, (int[]){ 20}), 20, "get test 1 dim");		
	
	tensor_reshape(a, 3, (int[]){3, 4, 5});
	
	// we test if the reshape works correct:
	expect(a->dims, 3, "reshape dims");
	expect(a->shape[0], 3, "reshape shape 0");
	expect(a->shape[1], 4, "reshape shape 1");
	expect(a->shape[2], 5, "reshape shape 2");
	expect(a->stride[0], 4*5, "reshape stride 0");
	expect(a->stride[1], 5, "reshape stride 1");
	expect(a->stride[2], 1, "reshape stride 2");
	
	int all_correct = 1;
	int value = 0;
	for(int i1=0; i1 < 3; i1++){
		for(int i2=0; i2 < 4; i2++){
			for(int i3=0; i3 < 5; i3++){
				if(tensor_get(a, (int[]){i1, i2, i3}) != value){
					all_correct = 0;
					expect(tensor_get(a, (int[]){i1, i2, i3}), value, "tensor get");
				}
				value++;
			}
		}
	}	
	
	// test one random position
	expect(tensor_get(a, (int[]) {2, 3, 4}), 2*4*5+3*5+4, "get test 3 dim");
	free_tensor(&a);
}

void test_mat_multiply(){
	tensor_handle_t* a = tensor_arange(0,12,1);
	tensor_handle_t* b = tensor_arange(0,12,1);
	int shape[] = {3, 4};
	tensor_reshape(a, 2, shape);
	
	shape[0] = 4;
	shape[1] = 3;
	tensor_reshape(b, 2, shape);
	
	tensor_handle_t* c = tensor_mat_multiply(a, b);	
	expect(c->data[0], 42, "test mat mult element [0]");
	expect(c->data[1], 48, "test mat mult element [1]");
	expect(c->data[2], 54, "test mat mult element [2]");
	expect(c->data[3], 114, "test mat mult element [3]");
	expect(c->data[4], 136, "test mat mult element [4]");
	expect(c->data[5], 158, "test mat mult element [5]");
	expect(c->data[6], 186, "test mat mult element [6]");
	expect(c->data[7], 224, "test mat mult element [7]");
	expect(c->data[8], 262, "test mat mult element [8]");
	free_tensor(&a);
	free_tensor(&b);
	free_tensor(&c);
}

void test_elm_add(){
	tensor_handle_t* a = tensor_random(3, (int[]){10, 2, 4});
	tensor_handle_t* b = tensor_arange(0, 4, 1);	
	tensor_reshape(b, 3, (int[]){1, 1, 4});
	tensor_elm_add(a, b);

	int all_correct = 1;
	for(int i1=0; i1 < 10; i1++){
		for(int i2=0; i2 < 2; i2++){
			for(int i3=0; i3 < 4; i3++){
				printf("v: %f", tensor_get(a, (int[]){i1, i2, i3}));
				if(tensor_get(a, (int[]){i1, i2, i3}) > i3 + 1 ||  tensor_get(a, (int[]){i1, i2, i3}) < i3){
					all_correct = 0;
					expect((int)tensor_get(a, (int[]){i1, i2, i3}), i3, "tensor elm add value should be in range v..v+1");
				}
			}
		}
	}
	expect(all_correct, 1, "test elm add with boradcasting");	
	print_tensor(a);
	free_tensor(&a);
	free_tensor(&b);
}

void test_layer(){
	//alter randmon sate
	//tensor_handle_t* lol = tensor_random(1,(int[]){10});
	//free_tensor(&lol);
	
	layer_t* layer1 = create_layer(2, 5);
	layer_t* layer2 = create_layer(5, 1);
	tensor_handle_t* input = tensor_arange(0,8,1); 
	tensor_reshape(input, 2, (int[]){4, 2});
	input->data[0] = 1;
	input->data[1] = 1;

	input->data[2] = 1;
	input->data[3] = -1;

	input->data[4] = -1;
	input->data[5] = -1;

	input->data[6] = -1;
	input->data[7] = 1;
	printf("input:\n");
	print_tensor(input);
	
	tensor_handle_t* y = create_tensor(2, (int[]) {4, 1});
	y->data[0] = 0;
	y->data[1] = 1;
	y->data[2] = 0;
	y->data[3] = 1;

	printf("y:\n");
	print_tensor(input);
	for(int epoch=0; epoch < 10; epoch++){	
		tensor_handle_t* output1 = forward_pass(layer1, input);
		tensor_handle_t* output = forward_pass(layer2, output1);
		print_tensor(output1);
		print_tensor(output);
		
		tensor_handle_t* error = squared_loss(output, y);
		float sum_error = tensor_sum(error);
		printf("loss: %f", sum_error);
		float learning_rate = 0.01;
		tensor_handle_t* error1 = backward_pass(layer2, error, learning_rate);
		tensor_handle_t* error0 = backward_pass(layer1, error1, learning_rate);
	
		printf("Done backward :)) \n\n\n\n");	

		free_tensor(&error);
		free_tensor(&error1);
		free_tensor(&error0);
		free_tensor(&output1);
		free_tensor(&output);
	}
	free_layer(&layer1);
	free_layer(&layer2);
	free_tensor(&input);
}

void test_transpose(){
	tensor_handle_t* test = tensor_arange(0, 12, 1);
	tensor_reshape(test, 2, (int[]){3, 4});
	print_tensor(test);
	tensor_transpose(test);
	print_tensor(test);
	// do actuall testing of some sorts. espacially the get_element function
	free_tensor(&test);
}

void test_loss(){
	// no error
	tensor_handle_t* output = tensor_arange(0, 10, 1);
	tensor_reshape(output, 2, (int[]){5, 2});
	printf("output:\n");
	print_tensor(output);
	tensor_handle_t* expected = tensor_copy(output);
	printf("expected:\n");
	print_tensor(expected);
	tensor_handle_t* error = squared_loss(output, expected);
	printf("error:\n");
	print_tensor(error);
	tensor_handle_t* zeros = create_tensor(2, error->shape);
	tensor_multiply(zeros, 0);
	expect(tensor_equal(error, zeros), 1, "expected error to be all zeros");
	free_tensor(&error);
	
	expected->data[0] = 0.25;
	expected->data[3] = 3.3;
	error = squared_loss(output, expected);
	printf("error2:\n");
	print_tensor(error);
	expect_f(error->data[0], 0.0625, "test squared loss");
	expect_f(error->data[3], 0.09, "test squared loss");
}

int main(){
	test_equal();
	test_creation();
	test_add();
	test_scalar_multiply();
	test_get();
	test_mat_multiply();
	test_elm_add();
	test_layer();
	test_transpose();
	test_loss();

	printf("%d tests. %d failed.\n", tests, failed);
}