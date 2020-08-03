
#ifndef CONNECT4_H
#define CONNECT4_H

#include <stdint.h>
#define ROWS 6

typedef struct c4_state{
	uint16_t field[ROWS]; //we use 2 bits for each position 10: stone by player 1 01: stone by player 0 and 00: free
	uint8_t player;
} c4_state_t;


uint8_t get_element(c4_state_t* state, uint8_t position, uint8_t row);
int check_win(c4_state_t* state, uint8_t position, uint8_t row);
int place(c4_state_t* state, uint8_t position);
int reverse(c4_state_t* state, uint8_t position);
void init_gameboard(c4_state_t* state);
void print_state(c4_state_t* state);
#endif /* CONNECT4_H */
