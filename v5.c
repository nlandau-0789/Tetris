/*

on représente le plateau dans un vec, avec chaque colonne comme un 16bit int, on a un big array avec les bitmasks (_m246i) pour le placement de chaque piece possible -> et les lignes qui sont possiblement supprimées dans ce cas

on passe le board en copie
pour supprimer les lignes completes, on fait des bitshift et des opérations binaires

_mm256_cmpge_epu16_mask

__builtin_popcount --> il faut -march=native
*/

#include "utils.h"
#define n_consts 12

// Utilitaires 
void print_board(__m256i board){
    short values[16];
    _mm256_storeu_si256((__m256i*)values, board);
    for (int y = 15; y >= 0; y--){
        printf("|");
        for (int x = 0; x < 10; x++){
            printf((values[x] & (1<<y)) ? "X":" ");
        }
        printf("|\n");
    }
    printf("+----------+\n");
}

__m256i board_from_array(bool *t){
    __m256i result = ZERO;

    unsigned short col0 = 0, col1 = 0, col2 = 0, col3 = 0, col4 = 0;
    unsigned short col5 = 0, col6 = 0, col7 = 0, col8 = 0, col9 = 0;

    for (int y = 15; y >= 0; y--) {
        col0 = (col0 << 1) | (t[y * 10 + 0] ? 1 : 0);
        col1 = (col1 << 1) | (t[y * 10 + 1] ? 1 : 0);
        col2 = (col2 << 1) | (t[y * 10 + 2] ? 1 : 0);
        col3 = (col3 << 1) | (t[y * 10 + 3] ? 1 : 0);
        col4 = (col4 << 1) | (t[y * 10 + 4] ? 1 : 0);
        col5 = (col5 << 1) | (t[y * 10 + 5] ? 1 : 0);
        col6 = (col6 << 1) | (t[y * 10 + 6] ? 1 : 0);
        col7 = (col7 << 1) | (t[y * 10 + 7] ? 1 : 0);
        col8 = (col8 << 1) | (t[y * 10 + 8] ? 1 : 0);
        col9 = (col9 << 1) | (t[y * 10 + 9] ? 1 : 0);
    }
    
    result = _mm256_set_epi16(0, 0, 0, 0, 0, 0, col9, col8, col7, col6, col5, col4, col3, col2, col1, col0);

    return result;
}

__m256i consts[n_consts];

void calc_consts(__m256i input) {
    __m256i heights = ZERO;
    __m256i holes = ZERO;
    __m256i cavities = ZERO;
    __m256i zero = ZERO;
    __m256i one = ONE;
    __m256i cav_mask = ONE;
    __m256i next_cav_mask = ZERO;
    __m256i two = _mm256_set1_epi16(2);

    for (int i = 0; i < 16; i++) {
        __m256i cmp = _mm256_and_si256(_mm256_cmpeq_epi16(input, zero), one);
        cmp = _mm256_subs_epu16(one, cmp);
        heights = _mm256_adds_epu16(heights, cmp);
        holes = _mm256_adds_epu16(holes, _mm256_subs_epu16(cmp, _mm256_and_si256(input, one)));
        next_cav_mask = _mm256_and_si256(input, one);
        cavities = _mm256_adds_epu16(cavities, _mm256_subs_epu16(next_cav_mask, cav_mask));
        cav_mask = next_cav_mask;
        input = _mm256_srli_epi16(input, 1);
    }
    

    __m256i rotated_right = rotate_right_one(heights);
    __m256i rotated_left = rotate_left_one(heights);
    __m256i max_height = bitonic_sort_epu16(heights);
    __m256i max_holes = bitonic_sort_epu16(holes);
    __m256i height_diffs = cut_epi16(_mm256_abs_epi16(_mm256_sub_epi16(rotated_left, heights)), 16-9);
    // __m256i sum_diffs = add_all_epi16(height_diffs);
    __m256i max_diff = bitonic_sort_epu16(height_diffs);
    __m256i max_cav = bitonic_sort_epu16(cavities);
    __m256i wells_deepness = cut_epi16(_mm256_min_epu16(_mm256_subs_epu16(rotated_right, heights), _mm256_subs_epu16(rotated_left, heights)), 16-10);
    __m256i wells_mask = _mm256_cmpgt_epi16(wells_deepness, two);
    __m256i wells = _mm256_and_si256(wells_mask, wells_deepness);
    __m256i is_well = _mm256_and_si256(wells_mask, one);

    consts[0] = heights;
    consts[1] = max_height;
    consts[2] = holes;
    consts[3] = max_holes;
    consts[4] = height_diffs;
    consts[5] = max_diff;
    consts[6] = cavities;
    consts[7] = max_cav;
    consts[8] = wells;
    consts[9] = is_well;
}

__m256i placements[7][4];
int n_rot[7] = {4, 2, 1, 4, 4, 2, 2}; // T, I, O, L, J, S, Z
int piece_width[7][4] = {
    {3, 3, 2, 2}, // T
    {4, 1, 0, 0}, // I
    {2, 0, 0, 0}, // O
    {3, 2, 3, 2}, // L
    {3, 3, 2, 2}, // J
    {3, 2, 0, 0}, // S
    {3, 2, 0, 0}  // Z
};
int piece_height[7][4] = {
    {2, 2, 3, 3}, // T
    {1, 4, 0, 0}, // I
    {2, 0, 0, 0}, // O
    {2, 3, 2, 3}, // L
    {2, 2, 3, 3}, // J
    {2, 3, 0, 0}, // S
    {2, 3, 0, 0}  // Z
};

void init_piece_placements(){
    bool *t = malloc(160 * sizeof(bool));
    for (int i = 0; i < 160; i++){
        t[i] = false;
    }

    // T
    t[10 * 15 + 0] = true;
    t[10 * 15 + 1] = true;
    t[10 * 15 + 2] = true;
    t[10 * 14 + 1] = true;
    placements[0][0] = board_from_array(t);
    // print_board(placementsnts[0][0]);
    t[10 * 15 + 0] = false;
    t[10 * 15 + 2] = false;
    t[10 * 14 + 0] = true;
    t[10 * 14 + 2] = true;
    placements[0][1] = board_from_array(t);
    // print_board(placementsnts[0][1]);
    t[10 * 15 + 1] = false;
    t[10 * 14 + 2] = false;
    t[10 * 15 + 0] = true;
    t[10 * 13 + 0] = true;
    placements[0][2] = board_from_array(t);
    // print_board(placementsnts[0][2]);
    t[10 * 15 + 1] = true;
    t[10 * 13 + 1] = true;
    t[10 * 15 + 0] = false;
    t[10 * 13 + 0] = false;
    placements[0][3] = board_from_array(t);
    // print_board(placementsnts[0][3]);
    
    t[10 * 15 + 1] = false;
    t[10 * 14 + 0] = false;
    t[10 * 14 + 1] = false;
    t[10 * 13 + 1] = false;

    // I
    t[10 * 15 + 0] = true;
    t[10 * 15 + 1] = true;
    t[10 * 15 + 2] = true;
    t[10 * 15 + 3] = true;
    placements[1][0] = board_from_array(t);
    // print_board(placementsnts[1][0]);
    t[10 * 15 + 1] = false;
    t[10 * 15 + 2] = false;
    t[10 * 15 + 3] = false;
    t[10 * 14 + 0] = true;
    t[10 * 13 + 0] = true;
    t[10 * 12 + 0] = true;
    placements[1][1] = board_from_array(t);
    // print_board(placementsnts[1][1]);
    t[10 * 15 + 0] = false;
    t[10 * 14 + 0] = false;
    t[10 * 13 + 0] = false;
    t[10 * 12 + 0] = false;

    // O
    t[10 * 15 + 0] = true;
    t[10 * 15 + 1] = true;
    t[10 * 14 + 0] = true;
    t[10 * 14 + 1] = true;
    placements[2][0] = board_from_array(t);
    // print_board(placementsnts[2][0]);
    t[10 * 15 + 0] = false;
    t[10 * 15 + 1] = false;
    t[10 * 14 + 0] = false;
    t[10 * 14 + 1] = false;

    // L
    t[10 * 15 + 0] = true;
    t[10 * 15 + 1] = true;
    t[10 * 15 + 2] = true;
    t[10 * 14 + 0] = true;
    placements[3][0] = board_from_array(t);
    // print_board(placementsnts[3][0]);
    t[10 * 15 + 1] = false;
    t[10 * 15 + 2] = false;
    t[10 * 13 + 0] = true;
    t[10 * 13 + 1] = true;
    placements[3][1] = board_from_array(t);
    // print_board(placementsnts[3][1]);
    t[10 * 15 + 0] = false;
    t[10 * 13 + 0] = false;
    t[10 * 13 + 1] = false;
    t[10 * 14 + 1] = true;
    t[10 * 14 + 2] = true;
    t[10 * 15 + 2] = true;
    placements[3][2] = board_from_array(t);
    // print_board(placementsnts[3][2]);
    t[10 * 15 + 0] = true;
    t[10 * 15 + 1] = true;
    t[10 * 15 + 2] = false;
    t[10 * 14 + 2] = false;
    t[10 * 14 + 0] = false;
    t[10 * 13 + 1] = true;
    placements[3][3] = board_from_array(t);
    // print_board(placementsnts[3][3]);

    t[10 * 15 + 0] = false;
    t[10 * 15 + 1] = false;
    t[10 * 14 + 1] = false;
    t[10 * 13 + 1] = false;

    // J
    t[10 * 15 + 0] = true;
    t[10 * 15 + 1] = true;
    t[10 * 15 + 2] = true;
    t[10 * 14 + 2] = true;
    placements[4][0] = board_from_array(t);
    // print_board(placementsnts[4][0]);
    t[10 * 15 + 1] = false;
    t[10 * 15 + 2] = false;
    t[10 * 14 + 0] = true;
    t[10 * 14 + 1] = true;
    placements[4][1] = board_from_array(t);
    // print_board(placementsnts[4][1]);
    t[10 * 14 + 1] = false;
    t[10 * 14 + 2] = false;
    t[10 * 15 + 1] = true;
    t[10 * 13 + 0] = true;
    placements[4][2] = board_from_array(t);
    // print_board(placementsnts[4][2]);
    t[10 * 15 + 0] = false;
    t[10 * 14 + 0] = false;
    t[10 * 14 + 1] = true;
    t[10 * 13 + 1] = true;
    placements[4][3] = board_from_array(t);
    // print_board(placementsnts[4][3]);

    t[10 * 15 + 1] = false;
    t[10 * 14 + 1] = false;
    t[10 * 13 + 0] = false;
    t[10 * 13 + 1] = false;

    // S
    t[10 * 15 + 1] = true;
    t[10 * 15 + 2] = true;
    t[10 * 14 + 0] = true;
    t[10 * 14 + 1] = true;
    placements[5][0] = board_from_array(t);
    // print_board(placementsnts[5][0]);
    t[10 * 15 + 1] = false;
    t[10 * 15 + 2] = false;
    t[10 * 15 + 0] = true;
    t[10 * 13 + 1] = true;
    placements[5][1] = board_from_array(t);
    // print_board(placementsnts[5][1]);

    t[10 * 15 + 0] = false;
    t[10 * 14 + 0] = false;
    t[10 * 14 + 1] = false;
    t[10 * 13 + 1] = false;

    // Z
    t[10 * 15 + 0] = true;
    t[10 * 15 + 1] = true;
    t[10 * 14 + 1] = true;
    t[10 * 14 + 2] = true;
    placements[6][0] = board_from_array(t);
    // print_board(placementsnts[6][0]);
    t[10 * 15 + 0] = false;
    t[10 * 14 + 2] = false;
    t[10 * 14 + 0] = true;
    t[10 * 13 + 0] = true;
    placements[6][1] = board_from_array(t);
    // print_board(placementsnts[6][1]);

    free(t);
}

struct placement_and_status {
    int x, r;
    float eval;
    bool dead;
};

typedef struct placement_and_status placement_and_status;

#include "nn.c"

void remove_full_lines(__m256i *board) {
    __m256i full_lines = *board;
    full_lines = _mm256_and_si256(full_lines, rotate_right(full_lines, 11));
    full_lines = _mm256_and_si256(full_lines, rotate_right(full_lines, 14));
    full_lines = _mm256_and_si256(full_lines, rotate_left_one(full_lines));
    full_lines = _mm256_and_si256(full_lines, rotate_left_one(full_lines));

    short full_lines_mask = _mm256_extract_epi16(full_lines, 0);
    for (int i = 15; i >= 0; i--) {
        if (full_lines_mask & (1 << i)) {
            // remove the line
            __m256i top = _mm256_slli_epi16(_mm256_srli_epi16(*board, i+1), i);
            // print_board(top);
            __m256i bottom = _mm256_srli_epi16(_mm256_slli_epi16(*board, 16-i), 16-i);
            // print_board(bottom);
            *board = _mm256_or_si256(top, bottom); // insert a new empty line at the top
        }
    }
}

__m256i place(int piece, int x, int r, __m256i board) {
    __m256i piece_board = placements[piece][r];
    for (int i = 0; i < x; i++) {
        piece_board = rotate_right_one(piece_board);
    }
    int y = 16;
    while (y > piece_height[piece][r] && is_zero_m256i(_mm256_and_si256(_mm256_srli_epi16(piece_board, 1), board))){ // potential +/-1 mistake
        y--;
        piece_board = _mm256_srli_epi16(piece_board, 1);
        // print_board(piece_board);
    }

    // evaluate the placement
    __m256i new_board = _mm256_or_si256(board, piece_board);
    remove_full_lines(&new_board);
    return new_board;
}

placement_and_status get_placement(int piece, __m256i board, nn *network) {
    float best_eval = 0.0f;
    placement_and_status best_placement = {0, 0, 0, true};
    for (int r = 0; r < n_rot[piece]; r++) {
        __m256i piece_board_init = placements[piece][r];
        for (int x = 0; x < 10-piece_width[piece][r]+1; x++) { // potential +/-1 mistake
            int y = 16;
            __m256i piece_board = piece_board_init;
            if (!is_zero_m256i(_mm256_and_si256(piece_board, board))){
                continue; // piece overlaps with board
            }
            while (y > piece_height[piece][r] && is_zero_m256i(_mm256_and_si256(_mm256_srli_epi16(piece_board, 1), board))){ // potential +/-1 mistake
                y--;
                piece_board = _mm256_srli_epi16(piece_board, 1);
            }

            // evaluate the placement
            __m256i new_board = _mm256_or_si256(board, piece_board);
            remove_full_lines(&new_board);
            calc_consts(new_board);

            // put consts in the nn input
            short input_short[160];
            for (int i = 0; i < 10; i++) {
                _mm256_storeu_si256((__m256i*)(&input_short[i*16]), consts[i]);
            }
            float input[160];
            for (int i = 0; i < 160; i++) {
                input[i] = (float)input_short[i];
            }

            // run the nn
            float *save_old_input = network->input;
            network->input = input;
            feed_forward(network, ReLU);
            float eval = network->output;
            network->input = save_old_input;

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

int train(nn *network, __m256i board, float learning_rate, int piece, int n_turns) {
    int next_piece = rand() % 7;
    // print_board(board);
    int max_depth = 0;
    for (int r = 0; r < n_rot[piece]; r++) {
        __m256i piece_board_init = placements[piece][r];
        for (int x = 0; x < 10-piece_width[piece][r]+1; x++) {
            printf("%d\r", n_turns);
            int y = 16;
            __m256i piece_board = piece_board_init;
            if (!is_zero_m256i(_mm256_and_si256(piece_board, board))){
                piece_board_init = rotate_right_one(piece_board_init);
                continue;
            }
            while (y > piece_height[piece][r] && is_zero_m256i(_mm256_and_si256(_mm256_srli_epi16(piece_board, 1), board))){
                y--;
                piece_board = _mm256_srli_epi16(piece_board, 1);
            }

            __m256i new_board = _mm256_or_si256(board, piece_board);
            remove_full_lines(&new_board);
            int depth = train(network, new_board, learning_rate, next_piece, n_turns + 1);

            calc_consts(new_board);
            short input_short[160];
            for (int i = 0; i < 10; i++) {
                _mm256_storeu_si256((__m256i*)(&input_short[i*16]), consts[i]);
            }
            float input[160];
            for (int i = 0; i < 160; i++) {
                input[i] = (float)input_short[i];
            }

            if (depth + 1 > max_depth) {
                max_depth = depth + 1;
            }
            
            piece_board_init = rotate_right_one(piece_board_init);
        }
    }
    if (max_depth > 3) {
        // no valid placements, game over
        calc_consts(board);
        short input_short[160];
        for (int i = 0; i < 10; i++) {
            _mm256_storeu_si256((__m256i*)(&input_short[i*16]), consts[i]);
        }
        float input[160];
        for (int i = 0; i < 160; i++) {
            input[i] = (float)input_short[i];
        }
        backpropagate(network, input, 1.0 - exp(-max_depth/50), ReLU, ReLU_derivative, learning_rate);
    }
    return max_depth;
}

int test_board() {
    // bool *t = aligned_alloc(32, 160 * sizeof(bool));
    bool *t = malloc(160 * sizeof(bool));
    if (!t) {
        perror("aligned_alloc failed");
        return 1;
    }
    for (int i = 0; i < 160; i++){
        t[i] = false;
    }
    for (int i = 0; i < 8; i++){
        t[i] = true;
    }
    for (int i = 2; i < 6; i++){
        t[10*i+3] = true;
    }
    for (int i = 4; i < 6; i++){
        t[10*i+4] = true;
    }
    for (int i = 0; i < 14; i+=2){
        t[10*i+8] = true;
    }
    for (int i = 0; i < 4; i++){
        t[10*i] = true;
    }
    for (int i = 0; i < 4; i++){
        t[10*i+2] = true;
    }
    t[159] = true;
    __m256i board = board_from_array(t);
    print_board(board);

    calc_consts(board);

    print_m256i_as_int16(consts[0]);
    print_m256i_as_int16(consts[1]);
    print_m256i_as_int16(consts[2]);
    print_m256i_as_int16(consts[3]);
    print_m256i_as_int16(consts[4]);
    print_m256i_as_int16(consts[5]);
    print_m256i_as_int16(consts[6]);
    print_m256i_as_int16(consts[7]);
    print_m256i_as_int16(consts[8]);
    print_m256i_as_int16(consts[9]);

    for (int i = 0; i < 10; i++) {
        t[10*3 + i] = true;
        t[10*4 + i] = true;
        t[10*7 + i] = true;
    }
    board = board_from_array(t);
    remove_full_lines(&board);
    print_board(board);

    return 0;
}

int main(){
    init_piece_placements();

    nn network;
    int n_hidden_layers = 3;
    int hidden_layer_sizes[] = {20, 20, 20};
    init_nn(&network, 160, n_hidden_layers, hidden_layer_sizes);


    train(&network, ZERO, 0.0001f, rand() % 7, 0);
}

