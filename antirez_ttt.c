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



// decision making (best_move, picking actions, update_q)


// pretraining (update_q, reward backpropogation)


// trigger loop (main game loop, training multiple games)