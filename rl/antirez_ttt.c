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

void backprop()

void learn_from_game()

void play_game()

int get_random_move()

char play_random_game()

void train_against_random()

int main()

