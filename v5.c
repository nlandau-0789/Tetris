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
    t[10 * 15 + 0] = false;
    t[10 * 15 + 2] = false;
    t[10 * 14 + 0] = true;
    t[10 * 14 + 2] = true;
    placements[0][1] = board_from_array(t);
    t[10 * 15 + 1] = false;
    t[10 * 14 + 2] = false;
    t[10 * 15 + 0] = true;
    t[10 * 13 + 0] = true;
    placements[0][2] = board_from_array(t);
    t[10 * 15 + 1] = true;
    t[10 * 13 + 1] = true;
    t[10 * 15 + 0] = false;
    t[10 * 13 + 0] = false;
    placements[0][3] = board_from_array(t);
    
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
    t[10 * 15 + 1] = false;
    t[10 * 15 + 2] = false;
    t[10 * 15 + 3] = false;
    t[10 * 14 + 0] = true;
    t[10 * 13 + 0] = true;
    t[10 * 12 + 0] = true;
    placements[1][1] = board_from_array(t);
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
    t[10 * 15 + 1] = false;
    t[10 * 15 + 2] = false;
    t[10 * 13 + 0] = true;
    t[10 * 13 + 1] = true;
    placements[3][1] = board_from_array(t);
    t[10 * 15 + 0] = false;
    t[10 * 13 + 0] = false;
    t[10 * 13 + 1] = false;
    t[10 * 14 + 1] = true;
    t[10 * 14 + 2] = true;
    t[10 * 15 + 2] = true;
    placements[3][2] = board_from_array(t);
    t[10 * 15 + 0] = true;
    t[10 * 15 + 1] = true;
    t[10 * 15 + 2] = false;
    t[10 * 14 + 2] = false;
    t[10 * 14 + 0] = false;
    t[10 * 13 + 1] = true;
    placements[3][3] = board_from_array(t);

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
    t[10 * 15 + 1] = false;
    t[10 * 15 + 2] = false;
    t[10 * 14 + 0] = true;
    t[10 * 14 + 1] = true;
    placements[4][1] = board_from_array(t);
    t[10 * 14 + 1] = false;
    t[10 * 14 + 2] = false;
    t[10 * 15 + 1] = true;
    t[10 * 13 + 0] = true;
    placements[4][2] = board_from_array(t);
    t[10 * 15 + 0] = false;
    t[10 * 14 + 0] = false;
    t[10 * 14 + 1] = true;
    t[10 * 13 + 1] = true;
    placements[4][3] = board_from_array(t);

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
    t[10 * 15 + 1] = false;
    t[10 * 15 + 2] = false;
    t[10 * 15 + 0] = true;
    t[10 * 13 + 1] = true;
    placements[5][1] = board_from_array(t);

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
    t[10 * 15 + 0] = false;
    t[10 * 14 + 2] = false;
    t[10 * 14 + 0] = true;
    t[10 * 13 + 0] = true;
    placements[6][1] = board_from_array(t);

    free(t);
}

int main(){
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


    init_piece_placements();
    return 0;
}