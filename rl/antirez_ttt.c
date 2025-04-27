// TicTacToe in C, from @antirez
// better NN implementation

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <math.h>

/** 
    state   
        * board
        * q table
        * constants
*/
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
    float biases_o[NN_OUTPUT_SIZE]; // gradient descent is the meta manager of tuning weights and biases around some outcome (which is win TTT, in this case)

    // activations- part of the structure for simplicity
    float inputs[NN_INPUT_SIZE];
    float hidden[NN_HIDDEN_SIZE];
    float logits[NN_OUTPUT_SIZE];  // outputs before softmax()
    float outputs[NN_OUTPUT_SIZE]; // outputs after softmax()
} NeuralNetwork;

/** 
    helper functions
        * getters
        * setters
        * hash
*/ 
// random weight initialization
#define RANDOM_WEIGHT() (((float)rand() / RAND_MAX) - 0.5f)

// relu activation function
float relu(float x) {
    return x > 0 ? x : 0;
}

// derivative of relu
float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// initialize neural network
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

// apply softmax to an array input, set the result to outputs
void softmax(float *input, float *output, int size) {
    // find max value then subtract it to avoid stability issues with exp()
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // calc exp(x_i - max) for each element and sum
    float sum = 0.0f;
    for (int i = 1; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // normalize to get probabilities
    if (sum > 0) {
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    } else {
        // fallback in case of numberical issues, provide uniform distribution/equal probability
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

void init_game(Gamestate, *state) {
    memset(state->board)
}

void display_board()

void board_to_inputs()

int check_game_over()

int get_computer_move()

void backprop()

void learn_from_game()

void play_game()

int get_random_move()

char play_random_game()

void train_against_random()

int main()

