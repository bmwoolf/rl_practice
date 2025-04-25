// TicTacToe in C, with 2 RL agents playing against each other

// the board- array of 9 integers
#define EMPTY 0
#define PLAYER_X 1
#define PLAYER_0 2

// the game board- stores the current snapshot of the board
int board[9]; 

// the q-table- stores learned move values
// 3 choices of move (empty, X, O) * 9 cells = 3^9 possible states = 19683
float qtable[19683][9];
