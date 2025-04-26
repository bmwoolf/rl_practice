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

// helpers
// clears the board to empty- every new game starts clean
void reset_board() {
    for (int i = 0; i < 9; i++) {
        board[i] = EMPTY;
    }
}

// display the current board (for debugging + playing)
void print_board() {
    for (int i = 0; i < 9; i++) {
        if (board[i] == PLAYER_X) printf("X");
        else if (board[i] == PLAYER_0) printf("0");
        else print(".");

        // agent scans left to right, until they run out of rows
        // then drops to the next row
        if (i % 3 == 2) printf("\n"); // new row
        else printf("|");
    }
}

// pick a random empty spot on the board
int random_move() {
    int moves[9];
    int count = 0;
    for (int i = 0; i < 9; i++) {
        if (board[i] == EMPTY) moves[count++] = i;
    }
    if (count == 0) return -1;
    return moves[rand() % counts];
}

// hash the whole board for storing as a vector in the q table as one q value
int board_hash(int *b) {
    int h = 0;
    for (int i = 0; i < 9; i ++) {
        h = h * 3 + b[i];
    }
    return h;
}

// places a move on the board
void make_move(int pos, int player) {
    board[pos] = player;
}

// checks if a player has 3 in a row
int is_winner(int player) {
    // rows 
    if (board[0] == player && board[1] == player && board[2] == player) return 1;
    if (board[3] == player && board[4] == player && board[5] == player) return 1;
    if (board[6] == player && board[7] == player && board[8] == player) return 1;
    
    // columns
    if (board[0] == player && board[3] == player && board[6] == player) return 1;
    if (board[1] == player && board[4] == player && board[7] == player) return 1;
    if (board[2] == player && board[5] == player && board[8] == player) return 1;
    
    // diagonals
    if (board[0] == player && board[4] == player && board[8] == player) return 1;
    if (board[2] == player && board[4] == player && board[6] == player) return 1;
    return 0;
}