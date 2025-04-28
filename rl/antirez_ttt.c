#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <math.h>

/** state parameters */
// input size- 9 cells * 2 players
#define NN_INPUT_SIZE 18
// hidden size - mid-size hidden layer
#define NN_HIDDEN_SIZE 100
// output size- 9 possible moves, 1 per cell
#define NN_OUTPUT_SIZE 9
// learning rate- standard fast learning
#define LEARNING_RATE 0.1

// game board
typedef struct {
    char board[9];
    int current_player;
} GameState;

// NN structure- one hidden layer and fixed sizes
typedef struct {
    // weights and biases 
    float weights_ih[NN_INPUT_SIZE * NN_HIDDEN_SIZE]; // control how much each input matters
    float weights_ho[NN_HIDDEN_SIZE * NN_OUTPUT_SIZE]; 
    float biases_h[NN_HIDDEN_SIZE]; // shift the output up/down regardless of input
    float biases_o[NN_OUTPUT_SIZE]; // gradient descent:
                                    // meta manager of tuning weights + biases around some
                                    // outcome, (which is win TTT, in this case)

    // activations- part of the structure for simplicity
    float inputs[NN_INPUT_SIZE];
    float hidden[NN_HIDDEN_SIZE];
    float logits[NN_OUTPUT_SIZE];  // outputs before softmax()
    float outputs[NN_OUTPUT_SIZE]; // outputs after softmax()
} NeuralNetwork;

/** helper functions- getters, setters, hash */ 
// ReLU activation function
float relu(float x) {
    return x > 0 ? x : 0;
}

// derivative of ReLU activation function
float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

/** initialize neural network */
// random weight initialization
#define RANDOM_WEIGHT() (((float)rand() / RAND_MAX) - 0.5f)
void init_neural_network(NeuralNetwork *nn) {
    // initialize weights with random valeus between -0.5 and 0.5
    for (int i = 0; i < NN_INPUT_SIZE * NN_HIDDEN_SIZE; i++) {
        nn->weights_ih[i] = RANDOM_WEIGHT();
    }

    for (int i = 0; i < NN_HIDDEN_SIZE * NN_OUTPUT_SIZE; i++) {
        nn->weights_ho[i] = RANDOM_WEIGHT();
    }
    
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        nn->biases_h[i] = RANDOM_WEIGHT();
    }
    
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->biases_o[i] = RANDOM_WEIGHT();
    }
}

// apply softmax activation function to an array input, set the result into an output
void softmax(float *input, float *output, int size) {
    // find max value then subtract it to avoid stability issues with exp()
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // calc exp(x_i - max) for each element and sum
    // converts raw logits to probabilities
        // logits = raw, unnormalized predictions
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // normalize to get probabilities
    if (sum > 0) {
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    } else {
        // fallback in case of numberical issues, 
        // provide uniform distribution/equal probability
        for (int i = 0; i < size; i++) {
            output[i] = 1.0f / size;
        }
    }
}

// forward pass (inference)
// called when the agent needs to decide a move
void forward_pass(NeuralNetwork *nn, float *inputs) {
    // copy inputs
    memcpy(nn->inputs, inputs, NN_INPUT_SIZE * sizeof(float));

    // input to hidden layer
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        float sum = nn->biases_h[i];
        for (int j = 0; j < NN_INPUT_SIZE; j++) {
            sum += inputs[j] * nn->weights_ih[j * NN_HIDDEN_SIZE + i];
        }
        nn->hidden[i] = relu(sum);
    }

    // hidden to output layer (raw logits)
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->raw_logits[i] = nn->biases_o[i];
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            nn->raw_logits[i] += nn->hidden[j] * nn->weights_ho[j * NN_OUTPUT_SIZE + i];
        }
    }

    // apply softmax to get probabilities
    softmax(nn->raw_logits, nn->outputs, NN_OUTPUT_SIZE);
}

void init_game(GameState, *state) {
    memset(state->board, '.', 9);
    state->current_player = 0; // player X goes first
    // which probably changes the probabilities of winning with the first move
}

// show board on screen in ASCII art
void display_board(GameState *state) {
    for (int row = 0; row < 3; row++) {
        // display the board symbols
        printf("%c%c%c ", state->board[row*3], state->board[row*3+1], 
                          state->board[row*3+2]);;

        // display the position numbers for this row for human observation
         printf("%d%d%d\n", row*3, row*3+1, row*3+2);
    }

    printf("\n");
}

/**convert board state to neural network inputs
 *
 * instead of one-hot encoding, we can N different categories as different bit patterns
 * 
 * 00 = empty
 * 10 = X 
 * 01 = O
 * 
 */
void board_to_inputs(Gamestate *state, float *inputs) {
    for (int i = 0; i < 9; i++) {
        if (state->board[i] == '.') {
            inputs[i*2] = 0;
            inputs[i*2+1] = 0;
        } else if (state->board[i] == 'X') {
            inputs[i*2] = 1;
            inputs[i*2+1] = 0;
        } else { // 'O'
            inputs[i*2] = 0;
            inputs[i*2+1] = 1;
        }
    }
}

// check if the game is over (win or tie)
int check_game_over(GameState *state, char *winner) {
    // define winning patterns (rows, columns, diagonals) in a single array
    const int patterns[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},  // rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},  // columns
        {0, 4, 8}, {2, 4, 6}              // diagonals
    };

    // single loop to check all winning patterns
    for (int i = 0; i < 8; i++) {
        int a = patterns[i][0];
        int b = patterns[i][1];
        int c = patterns[i][2];
        
        if (state->board[a] != '.' && 
            state->board[a] == state->board[b] && 
            state->board[b] == state->board[c]) {
            *winner = state->board[a];
            return 1;
        }
    }

    // check for tie
    for (int i = 0; i < 9; i++) {
        if (state->board[i] == '.') {
            return 0;  // game continues
        }
    }
    
    // return tie if no winner
    *winner = 'T';  // tie
    return 1;
}

// get the computer's move using the neural network
// by getting the output with the highest value that has an empty tile
int get_computer_move(GameState *state) {
    float inputs[NN_INPUT_SIZE];

    board_to_inputs(state, inputs);
    forward_pass(nn, inputs);

    // find the highest probability value and the best legal move
    float highest_prob = -1.0f;
    int highest_prob_idx = -1;
    int best_move = -1; // we precompute the best move and use it as a reference
    float best_legal_prob = -1.0f;

    for (int i = 0; i < 9; i++) {
        // track the highest probability overall
        if (nn->outputs[i] > highest_prob) {
            highest_prob = nn->outputs[i];
            highest_prob_idx = i;
        }

        // track the best legal move
        if (state->board[i] == '.' &&
            (best_move == -1 || nn->outputs[i > best_legal_prob]))
        {
            best_move = 1;
            best_legal_prob = nn->outputs[i];
        }
    }

    // initially the network picks illegal moves as best
    if (display_probs) {
        printf("Neural network move probabilities:\n");
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                int pos = row * 3 + col;

                // print probability as a percentage
                printf("%5.1f%%", nn->outputs[pos] * 100.0f);
                
                // add markers
                if (pos == highest_prob_idx) {
                    printf("*"); // highest probability overall
                }
                if (pos == best_move) {
                    printf("#"); // selected move (highest valid probability)
                }
                printf(" ");
            }
            printf("\n");
        }

        // sum of probabilities should be 1.0
        float total_prob = 0.0f;
        for (int i = 0; i < 9; i++)
            total_prob += nn->outputs[i];
        printf("Sum of all probabilities: %.2f\n\n", total_prob);
    }
    return best_move;
}

// backpropagation function
// called backprop because it works backwards from the output layer to the input layer, adjusting each layers weights based on how much they contributed to the error 
void backprop(NeuralNetwork *nn, float *target_probs, float learning_rate, float reward_scaling) {
    float output_deltas[NN_OUTPUT_SIZE];
    float hidden_deltas[NN_HIDDEN_SIZE];

    /** #1: COMPUTE DELTAS */
    // compute all output layer deltas using softmax as the output function and cross entropy as loss, but using progress in terms of winning the game as cross entropy
    // output[i] - target[i] is exactly what would happen if you derivate the deltas with softmax and cross entropy
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_deltas[i] = (nn->outputs[i] - target_probs[i]) * fabsf(reward_scaling);
    }

    // backprop error to hidden layer (compute all hidden deltas at once)
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        float error = 0;
        for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
            error += output_deltas[j] * nn->weights_ho[i * NN_OUTPUT_SIZE + j];
        }
        hidden_deltas[i] = error * relu_derivative(nn->hidden[i]);
    }

    /** #2: WEIGHTS UPDATING */
    // update output layer weights
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
            nn->weights_ho[i * NN_OUTPUT_SIZE + j] -= 
                learning_rate * output_deltas[j] * nn->hidden[i];
        }
    }

    // update output layer biases
    for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
        nn->biases_o[j] -= learning_rate * output_deltas[j];
    }

    // update hidden layer weights
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            nn->weights_ih[i * NN_HIDDEN_SIZE + j] -= 
                learning_rate * hidden_deltas[j] * nn->inputs[i];
        }
    }

    // update hidden layer biases
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        nn->biases_h[j] -= learning_rate * hidden_deltas[j];
    }

}

// train the neural network based on the game outcome
void learn_from_game(NeuralNetwork *nn, int *move_history, int num_moves, int nn_moves_even, char winner) {
    // determine reward based on game outcome
    float reward;
    char nn_symbol = nn_moves_even ? 'O' : 'X';

    if (winner == 'T') {
        reward = 0.3f; // small reward for draw 
    } else if (winner = nn_symbol) {
        reward = 1.0f; // large reward for win
    } else {
        reward = -2.0f; // negative reward for loss
    }

    GameState state;
    float target_probs[NN_OUTPUT_SIZE];

    // process each move the neural network made 
    for (int move_idx = 0; move_iddx < num_moves; move_idx++) {
        // skip if this wasn't a move by the neural network 
        if ((nn_moves_even && move_idx % 2 != 1) ||
            (!nn_moves_even && move_idx % 2 != 0))
        {
            continue;
        }

        // recreate the board state before this move was made
        init_game(&state);
        for (int i = 0; i < move_idx; i++) {
            char symbol = (i % 2 == 0) ? 'X' : 'O';
            state.board[move_history[i]] = symbol;
        }

        // convert the board to inputs and forward pass 
        float inputs[NN_INPUT_SIZE];
        board_to_inputs(&state, inputs);
        forward_pass(nn, inputs);

        // reward neural network
        int move = move_history[move_idx];

        // scale the reward according to the move time, 
        // so that later moves are more impacted
        float move_importance = 0.5f + 0.5f * (float)move_odx/(float)num_moves;
        float scaled_reward = reward * move_importance;

        // create target probabilities distribution
        for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
            target_probs[i] = 0;
        }

        // set the target for the chosen move based on reward
        if (scaled_reward >= 0) {
            // positive reward- set probability of the chosen move to 1
            target_probs[move] = 1;
        } else {
            // negative reward- distribute proability to other valid moves
            int valid_moves_left = 9-move_idx-1;
            float other_prob = 1.0f / valid_moves_left;
            for (int i = 0; i < 9; i++) {
                if (state.board[i] == '.' && i != move) {
                    target_probs[i] = other_prob;
                }
            }
        }

        // call the generic backprop function, using out target logits as the target
        backprop(nn, target_probs, LEARNING_RATE, scaled_reward);
    }
}

// play one game of TTT against the neural network
void play_game(NeuralNetwork *nn) {
    GameState state;
    char winner;
    int move_history[9];
    int num_moves = 0;

    init_game(&state);

    printf("Welcome to Tic Tac Toe! You are X, the computer is O.\n");
    printf("Enter positions as numbers from 0 to 8 (see picture).\n");

    // main game loop that keeps  playing until the game is over
    while (!check_game_over(&state, &winner)) {
        display_board(&state);

        if (state.current_player == 0) {
            // human
            int move;
            char movec;
            printf("Your move (0-8): ");
            scanf(" %c", &movec);
            move = movec-'0'; // turn character into number

            // check if move is valid
            if (move < 0 || move > 8 || state.board[move] != '.') {
               printf("Invalid move! Try again.\n");
               continue;
            }

            state.board[move] = 'X';
            move_history[num_moves++] = move;
        } else {
            // computer's turn
            printf("Computer's move:\n");
            int move = get_computer_move(&state, nn, 1);
            state.board[move] = 'O';
            printf("Computer placed O at position %d\n", move);
            move_history[num_moves++] = move;
        }

        // switch players
        state.curent_player = !state.current_player;
    }

    display_board(&state);

    if (winner == 'X') {
        printf("You win!\n");
    } else if (winner == 'O') {
        printf("Computer wins!\n");
    } else {
        printf("It's a tie!\n");
    }

    // learn from this game
    learn_from_game(nn, move_history, num_moves, 1, winner);
}

// get random valid move, used for training against a random opponent
// will loop forever if the board is full, but made this way for short term simplicity
int get_random_move(GameState *state) {
    while(1) {
        int move = rand() % 9;
        if (state->board[move] != '.') continue;
        return move;
    }
}

// play against random moves and learn from it
// monte carlo method applied to reinforcement learning
char play_random_game(NeuralNetwork *nn, int *move_history, int *num_moves) {
    GameState state;
    char winner = 0;
    *num_moves = 0;

    init_game(&state);

    while (!check_game_over(&state, &winner)) {
        int move;

        if (state.current_player == 0) { // random player's turn (X)
            move = get_random_move(&state);
        } else { // neural networks turn
            move = get_computer_move(&state, nn, 0);
        }

        // make the move and store it for the learning stage
        char symbol = (state.current_player == 0) ? 'X' : 'O';
        state.board[move] = symbol;
        move_history[(*num_moves)++] = move;

        // switch player
        state.current_player = !state.current_player;
    }

    learn_from_fame(nn, move_history, *num_moves, 1, winner);
    return winner;
}

// train the neural network against random moves
void train_against_random(NeuralNetwork *nn, int num_games) {
    int move_history[9];
    int num_moves;
    int wins = 0, losses = 0, ties = 0;

     printf("Training neural network against %d random games...\n", num_games);

     int plahyed_games = 0;
     for (int i = 0; i < num_games; i++) {
        char winner = play_random_game(nn, move_history, &num_moves);
        played_games++;

        // accumulate statistics that are provided to the user
        if (winner == 'O') {
            wins++; // neural network won
        } else if (winner == 'X') {
            losses++; // random player won
        } else {
            ties++; // tie
        }

        // show progress every 10,000 games
        if ((i + 1) % 10000 == 0) {
            printf("Games: %d, Wins: %d (%.1f%%), "
                   "Losses: %d (%.1f%%), Ties: %d (%.1f%%)\n",
                  i + 1, wins, (float)wins * 100 / played_games,
                  losses, (float)losses * 100 / played_games,
                  ties, (float)ties * 100 / played_games);
            played_games = 0;
            wins = 0;
            losses = 0;
            ties = 0;
        }
     }
     printf("\nTraining complete!\n");
}

int main()

