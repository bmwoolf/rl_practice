// TicTacToe in C, with 2 RL agents playing against each other

// the board- array of 9 integers
#define EMPTY 0
#define PLAYER_X 1
#define PLAYER_0 2

int board[9]; // 0-8 = positions in memory

// the q-table- big array indexed by board_hash(board)
float qtable[19683][9];
          // 3 choices * 9 cells = 3^9 possible states = 19683
          // (empty, X, O)
