// connect 4 c implementation.

#include "connect4.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

uint8_t get_element(c4_state_t* state, uint8_t position, uint8_t row){
	return (state->field[row] >> (position*2)) & 3;
}

int check_win(c4_state_t* state, uint8_t position, uint8_t row){
	// check for horizontal
	uint16_t mask = (1 + (1 << 2) + (1 << 4) + (1 << 6)) << state->player;
	
	// check left side
	uint8_t left_beginning = 0;
	uint8_t right_beginning = 3;
	/*if(position > 3){
		left_beginning = position - 3;
	}
	// check right side
	if(position + 3 < 7){
		right_beginning = position;	
		// 0 1 2 3 4 5 6
		//       x x x x
	}*/
	//printf("checking between %d and %d", left_beginning, right_beginning);
	for(uint8_t check_position = left_beginning; check_position <= right_beginning; check_position++){	
		uint16_t check_mask = mask << (check_position * 2);
		if((state->field[row] & check_mask) == check_mask){
			//printf("horisontal win at: %d\n", check_position);
			return 2; // won
		}
	}
	// check horizontal
	uint8_t up_beginning = ROWS;
	if(row < ROWS-4){
		up_beginning = row + 3;
	}
	uint8_t down_beginning = 0;
	if(row > 3){
		down_beginning = row - 3;	
	}
	
	//we go outward in every direction and count the length of elements of that one player.
	int8_t lengthes[3][2];	
	for(int8_t d_p = -1 ; d_p <= 1; d_p++){
		for(int8_t d_r = -1 ; d_r <= 1; d_r+=2){
			//check if still in bounds
			uint8_t c_position = position + d_p;
			uint8_t c_row = row + d_r;
			uint8_t length = 0;	
			
			while(length < 3
			      && c_position >= 0
			      && c_position < 7
			      && c_row >= 0
			      && c_row < ROWS
			      && get_element(state, c_position, c_row) == (1 << state->player)){
				c_position += d_p;
				c_row += d_r;
				length++;
			}
			//printf("d_p: %d d_r: %d l:%d\n", d_p, d_r, length);
			lengthes[d_p + 1][(d_r+1)/2] = length;
		}	
	}
	// 0 : / [0][0] + [2][1]
	// 1 : | [1][0] + [1][1]
	// 2 : \ [2][0] + [0][1]
	if(lengthes[0][0] + lengthes[2][1] + 1 >= 4){
		//printf("i: 0(/) length is sufficent\n");
		return 2; //won
	}
	if(lengthes[1][0] + lengthes[1][1] + 1 >= 4){
		//printf("i: 1(|) length is sufficent\n");
		return 2; //won
	}
	if(lengthes[2][0] + lengthes[0][1] + 1 >= 4){
		//printf("i: 2(\\) length is sufficent\n");
		return 2; //won
	}
	// check if its a draw
	if(row == ROWS -1){
		//printf("check last:\n");
		//printf("%x\n", ((state->field[row] & 0x1555) << 1));
		//printf("%x\n", (state->field[row] & (0x1555 << 1)));
		//printf("%x\n",(((state->field[row] & 0x1555) << 1) | (state->field[row] & (0x1555 << 1))));
		// we fist select the bits of player 0. Shift them to the left and or them with the bits of player 1.
		// then check if all the 7 bits are one.
		if((((state->field[row] & 0x1555) << 1) | (state->field[row] & (0x1555 << 1))) == (0x1555 << 1)){
			return 3;
			printf("DRAW\n");
		}
	}
	return 1; // not won
}

int8_t get_last_stone(c4_state_t* state, uint8_t position){
	// go down till we find empty position
	uint16_t mask = 3 << (position * 2);
	int8_t last_stone = -1;
	for(int row = ROWS - 1; row >= 0; row--){
		if(state->field[row] & mask){
			last_stone = row;
			break;
		}
	}	
	return last_stone;
}

int place(c4_state_t* state, uint8_t position){
	if(position > 7 || position < 0){
		printf("position needs to be in between 0 and 7 and is %d", position);
		return -2;
	}
	int8_t last_stone = get_last_stone(state, position);
	
	// check if ther is still space
	if(last_stone == ROWS - 1){
		return -1;
	}
	// place it in row last_stone + 1;	
	state->field[last_stone + 1] |= 1 << (state->player + 2 * position);
	//printf("placing stone in position:%d\n", last_stone + 1);
	
	// check if the player won	
	int win_state = check_win(state, position, last_stone + 1);
	// switch player
	// we always switch the player if the player won. the other player is the actual player that won...
	// this is for convidience in the reverse move function.
	//if(win_state != 2){
		state->player ^= 1; 	
	//}
	return win_state;
}

int reverse(c4_state_t* state, uint8_t position){
	int8_t last_stone = get_last_stone(state, position);
	if(last_stone == -1){
		printf("ERROR: can't reverse move on position %d playing_field:\n", position);
		print_state(state);
	}
	// set the two bits to zero
	state->field[last_stone] &= ~(3 << position*2);
	// switch player
	state->player ^= 1; 	
	return 1;
}

void init_gameboard(c4_state_t* state){
	for(uint8_t i = 0; i < ROWS; i++){
		state->field[i] = 0;
	}
	state->player = 0;
}

void print_state(c4_state_t* state){
	for(int8_t r_i = ROWS - 1; r_i >= 0; r_i--){
		//printf("%x\n", (unsigned char)state->field[r_i]);
		for(uint8_t p_i = 0; p_i < 7; p_i++){
			int8_t elm = (uint8_t)get_element(state, p_i, r_i);
			//if(elm == 2){
		//		elm = -1;
			//}
			printf("%d ", elm);
		}
		printf("\n");
	}
	printf("-------------\n");
	printf("0 1 2 3 4 5 6\n");
	printf("player: %d\n", state->player);
}
/*
main to play manual or for performance benchmark

int main(){
	// create gameboard
	c4_state_t state;
	int steps = 0;
	clock_t  begin_a = clock();
	for(int i=0; i < 100000; i++){
		int last_state = 0;
		init_gameboard(&state);	
		//state.field[ROWS - 1] = 0x0665;
		//state.field[ROWS - 2] = 0x1000;
		while(!(last_state == 2 || last_state == 3)){
			//print_state(&state);
			*//*
			printf("input:");
			char c = EOF;
			while(c == EOF || c == '\n'){
				c = getchar();
			}
			if(c == 'x'){
				printf("ok exiting");
				break;
			}	
			printf("Character entered: ");
			putchar(c);
			printf("\n");
			int8_t position = c - '0';
			*/
			/*
			int8_t position = rand() % 7;
			//printf("try input %d\n", position);
			clock_t begin = clock();
			last_state = place(&state, position);
			steps++;
			clock_t end = clock();
			//printf("time: %lu", end - begin);
			//printf("game state: %d\n", last_state);
		}
	}
	printf("steps: %d\n", steps);
	printf("time: %f\n", (clock() - begin_a) / (double)CLOCKS_PER_SEC);
}
*/

