#include "v5.c"

struct creature {
    __m256i *weights;
    int n_played;
    double total_score;
};

typedef struct creature creature;

// lazy segtree pour trouver le max UCB score

double ask_creature(creature *c, __m256i board){
    calc_consts(board);
    int total = 0;
    __m256i sum = ZERO; 
    for (int i = 0; i < N_FEATURES; i++){
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(consts[i], c->weights[i]));
    }

}

placement get_placement(int piece, __m256i board, creature *c) {
    float best_eval = -INFINITY;
    placement best_placement = {0, 0, 0, true};
    for (int r = 0; r < n_rot[piece]; r++) {
        __m256i piece_board_init = placements[piece][r];
        for (int x = 0; x < 10-piece_width[piece][r]+1; x++) { // potential +/-1 mistake
            #ifdef DEBUG_VERBOSE
            // printf("Trying piece %d, rotation %d, x=%d\n", piece, r, x);
            #endif
            int y = 16;
            __m256i piece_board = piece_board_init;
            if (!is_zero_m256i(_mm256_and_si256(piece_board, board))){
                #ifdef DEBUG_VERBOSE
                fprint_board(log_file, board);
                fprint_board(log_file, piece_board);
                fprintf(log_file, "Piece overlaps with board at x=%d, r=%d\n", x, r);
                #endif
                piece_board_init = rotate_right_one(piece_board_init);
                continue; // piece overlaps with board
            }
            while (y > piece_height[piece][r] && is_zero_m256i(_mm256_and_si256(_mm256_srli_epi16(piece_board, 1), board))){ // potential +/-1 mistake
                y--;
                piece_board = _mm256_srli_epi16(piece_board, 1);
                #ifdef DEBUG_VERBOSE
                fprint_board(log_file, piece_board);
                #endif
            }

            // evaluate the placement
            __m256i new_board = _mm256_or_si256(board, piece_board);
            remove_full_lines(&new_board);

            double eval = ask_creature(c, new_board);

            if (eval > best_eval) {
                best_eval = eval;
                best_placement.x = x;
                best_placement.r = r;
                best_placement.eval = eval;
                best_placement.dead = false;
            }
            
            piece_board_init = rotate_right_one(piece_board_init);
        }
    }
    return best_placement;
}

void sample(creature *c){
    __m256i board = ZERO;
    int piece = randint(0, 7);
    int n_lines_removed = 0;

    for (int turn = 0; turn < 2000000000; turn++) {
        placement placement = get_placement(piece, board, network);
        #ifdef DEBUG_VERBOSE
        fflush(stdout);
        printf("%f\n", placement.eval);
        #endif
        if (placement.dead) {
            // printf("Game over! Dead at turn %d\n", turn);
            return turn;
        }
        board = place(piece, placement.x, placement.r, board, &n_lines_removed);
        #ifdef DEBUG_VERBOSE
        print_board(board);
        #endif
        piece = rand() % 7;
    }
    return 2000000000;
}