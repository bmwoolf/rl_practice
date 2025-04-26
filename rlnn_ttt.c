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

// check if the board is full, but no one won
int is_draw() {
    // scan all 9 cells
    for (int i = 0; i < 9; i++) {
        // if any are empty, no draw yet -> return 0
        if (board[i] == EMPTY) return 0; // still moves left
    }
    // if none are empty, return 1 for draw
    return 1;
}

// RL logic
// ai picks a move
int select_move(int player) {
    if ((rand() % 100) < 20) {
        // 20% chance- random move 
        return random_move();
    }

    // 80% chance- pick the best move based on the q table
    int best_move = -1;
    float best_q = -1e9; // very small number to start
    int h = board_hash(board);

    for (int i = 0; i < 9; i++) {
        if (board[i] == EMPTY) {
            if (qtable[h][i] > best_q) {
                best_q = qtable[h][i];
                best_move = i;
            }
        }
    }

    // fallback: if somehow no best move found, pick random
    if (best_move == -1) return random_move()
    return best_move;
}