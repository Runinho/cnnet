#include "connect4.h"
#include "tensor.h"
#include "model.h"

#include <stdlib.h>
#include <stdio.h>


tensor_handle_t* create_gamestate_tensor(int num_state){
	return create_tensor(2, (int[]) {num_state, ROWS * 7});
}

void gamestate_to_tensor(c4_state_t* state, int position, tensor_handle_t* t){
	int num_elements = ROWS * 7;
	int element_i = 0;
	for(int c_r=0; c_r < ROWS; c_r++){
		uint16_t row_data = state->field[c_r];
		for(int c_p=0; c_p < 7; c_p++){
			for(int c_player=0; c_player < 2; c_player++){
				*tensor_get_p(t, (int[]){position, element_i}) = (float)(row_data & 1);
				row_data = row_data >> 1;
				element_i++;
			}
		}
	}
	printf("element_i: %d\n", element_i);
	if(t->shape[1] != element_i){
		printf("WARNING: saving gamestate_i in a row with %d data. but need %di\n", t->shape[1], element_i);
	}
}


int pridict_scores(c4_state_t* game_state, tensor_handle_t* out_scores){
	int is_valid[7];
	tensor_handle_t* eval = create_tensor(2, (int[]){7, 42});
	int max_score_i = predict_scores_(game_state, is_valid, eval, out_scores);
	free_tensor(&eval);
	return max_score_i;
}

int predict_scores_(c4_state_t* game_state, int* is_valid, tensor_handle_t* eval, tensor_handle_t* out_scores){
	// use this method only if you wanr reuse is_valid and eval. Otherwise use `predict_scores`
	// predict the scores for one gamestate
	// is_valid needs to be and int[7]
	// eval an tensor with shape [7, 42]
	// out_scores is a tensor with shape [7, 1];
	for(uint8_t c_p=0; c_p < 7; c_p++){
		int tmp_game_state = place(&game_state, c_p);
		if(tmp_game_state > 0){
			//move is valid;
			is_valid[c_p] = 1;
			// write the current game state into the model input;	
			gamestate_to_tensor(&game_state, c_p, eval);
			//printf("updated gamestate tensor:\n");
			//print_tensor(tmp_eval);
			// reverse the move
			reverse(&game_state, c_p);
		} else {
			is_valid[c_p] = 0;
		}
	}
	//printf("predicting\n");	
	// do a forward pass to get the current predicted scores.
	tensor_handle_t* scores = predict(models[c_player], tmp_eval);	
	
	// find max score.
	int max_score_i = -1;
	for(uint8_t c_p=0; c_p < 7; c_p++){
		if(is_valid[c_p] 
		   && (max_score_i == -1 
		       || tensor_get(scores, (int[]) {max_score_i, 1}) < tensor_get(scores, (int[]) {c_p, 1}))){
			max_score_i = c_p;
		}
	}

	return max_score_i;
}

void train_game(){
	// setup gameboard
	c4_state_t game_state;
	
	model_t* models[2];	
	models[0] = sequential(5, (int[]) {42, 128, 128, 128, 1}, sigmoid);
	models[1] = sequential(5, (int[]) {42, 128, 128, 128, 1}, sigmoid);
	
	// attack plan:
	// play one game: predict -> choose(with epsilon random) -> 
	// save states and predicted value. Fixed size array with size 42 (max number of moved)
	// update weigts from end result of game backwards. 
	// 	So Winner gets 1. next state is + d * 1. and so forth...
	//	Loser gets -1 and all state before get - d * -1

	// initalize buffers
	tensor_handle_t* states[2];
	states[0] = create_tensor(2, (int[]){21, 42});	
	states[1] = create_tensor(2, (int[]){21, 42});	
	tensor_handle_t* predictions[2];
	predictions[0] = create_tensor(2, (int[]){21, 1});	
	predictions[1] = create_tensor(2, (int[]){21, 1});	
	tensor_handle_t* tmp_eval = create_tensor(2, (int[]){7, 42});	
	
	printf("done setup.\n");
	
	for(int games_i = 0; games_i < 10000; games_i++){
	init_gameboard(&game_state);	

	int last_game_state = 0;
	int game_steps[2] = {0, 0};

	// while game not won or draw.
	while(last_game_state < 2){
		int is_valid[7];
		int c_player = game_state.player;
		// go through all positions and get the predicted value;
		for(uint8_t c_p=0; c_p < 7; c_p++){
			int tmp_game_state = place(&game_state, c_p);
			if(tmp_game_state > 0){
				//move is valid;
				is_valid[c_p] = 1;
				// write the current game state into the model input;	
				gamestate_to_tensor(&game_state, c_p, tmp_eval);
				//printf("updated gamestate tensor:\n");
				//print_tensor(tmp_eval);
				// reverse the move
				reverse(&game_state, c_p);
			} else {
				is_valid[c_p] = 0;
			}
		}
		//printf("predicting\n");	
		// do a forward pass to get the current predicted scores.
		tensor_handle_t* scores = predict(models[c_player], tmp_eval);	
		
		// find max score.
		int max_score_i = -1;
		for(uint8_t c_p=0; c_p < 7; c_p++){
			if(is_valid[c_p] 
			   && (max_score_i == -1 
			       || tensor_get(scores, (int[]) {max_score_i, 1}) < tensor_get(scores, (int[]) {c_p, 1}))){
				max_score_i = c_p;
			}
		}
		if(max_score_i == -1){
			printf("ERROR: THIS SHOULDN'T happen couldn't find a valid move. is the draw detection working correctly???\n");
			//free all the stuff
			free_model(&models[0]);
			free_model(&models[1]);
			free_tensor(&scores);
			free_tensor(&tmp_eval);
			free_tensor(&states[0]);
			free_tensor(&states[1]);
			free_tensor(&predictions[0]);
			free_tensor(&predictions[1]);
			return;
		}
		// add random try. (easyest: choose random number check if valid. and then use that move..))
		int choosed_random = 0;
		if(random() % 10 == 0){
			// we try to choose a random one.
			int try_pos = random() % 7;
			if(is_valid[try_pos]){
				max_score_i = try_pos;
				choosed_random = 1;
			}
		}
		
		// do the choosen move.
		last_game_state = place(&game_state, max_score_i);	
		//printf("playing %d, random: %d\n", max_score_i, choosed_random);
		//print_state(&game_state);

		// save the choosen move to positon tensor as well as the predicted value.
		*tensor_get_p(predictions[c_player], (int[]){game_steps[c_player], 0}) = tensor_get(scores, (int[]) {max_score_i, 0});
		gamestate_to_tensor(&game_state, game_steps[c_player], states[c_player]);
		game_steps[c_player]++;
		
		// free tensors
		free_tensor(&scores);
	}
	// we should have a draw or win. Update the scores acordingly and train both models for one epoch
	float losses[2];
	if(last_game_state == 3){
		// we have a draw;
		losses[0] = -0.5;
		losses[1] = -0.5;
	}
	if(last_game_state == 2){
		// one player won the game;
		uint8_t player_won = 1 ^ game_state.player;
		losses[player_won] = 1;
		losses[game_state.player] = -1;
	}
	// we update the scores
	float delta = 0.2;
	float learning_rate = 0.01;
	if(games_i % 100 == 0){
		printf("game %d  with %d steps\n", games_i, game_steps[0]);
	}
	//printf("updating scores\n");	
	for(int c_p = 0; c_p < 2; c_p++){
		//last step is fixed.
		*tensor_get_p(predictions[c_p], (int[]) {game_steps[c_p] - 1, 0}) = losses[c_p];
		// update the other steps
		for(int i=game_steps[c_p]-2; i >=0; i--){
			float tmp = tensor_get(predictions[c_p], (int[]) {i, 0}) + delta * losses[c_p];
			losses[c_p] = tmp;
			*tensor_get_p(predictions[c_p], (int[]) {i, 0}) = losses[c_p];
		}
		states[c_p]->shape[0] = game_steps[c_p];
		predictions[c_p]->shape[0] = game_steps[c_p];
		
		//printf("states:\n");
		//print_tensor(states[c_p]);
		//printf("predictions:\n");
		////print_tensor(predictions[c_p]);
		
		//printf("training\n");
		float* history = train(models[c_p], states[c_p], predictions[c_p], 1, learning_rate);
		//printf("end training\n");
		free(history);	
	}
	}
	
	//saving model_weights
	save_weights(models[0], "model_p1");
	save_weights(models[1], "model_p2");
	//free all the stuff
	free_model(&models[0]);
	free_model(&models[1]);
	free_tensor(&tmp_eval);
	free_tensor(&states[0]);
	free_tensor(&states[1]);
	free_tensor(&predictions[0]);
	free_tensor(&predictions[1]);
	return;
}


int main(){
	train_game();
	printf("done.");
}
