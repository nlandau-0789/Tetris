/*

on représente le plateau dans un vec, avec chaque colonne comme un 16bit int, on a un big array avec les bitmasks (_m246i) pour le placement de chaque piece possible -> et les lignes qui sont possiblement supprimées dans ce cas

on passe le board en copie
pour supprimer les lignes completes, on fait des bitshift et des opérations binaires

_mm256_cmpge_epu16_mask

__builtin_popcount --> il faut -march=native
*/

#include <immintrin.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#define n_consts 12


// Utilitaires 

#define ZERO _mm256_set1_epi16(0x0000)
#define ONES _mm256_set1_epi16(0xFFFF)
#define ONE _mm256_set1_epi16(0x0001)

__m256i shift_right_one(__m256i vec) {
    // Shift the last element to the first position
    __m256i last_element = _mm256_permute2x128_si256(vec, vec, 0x21); // Swap the 128-bit lanes
    last_element = _mm256_srli_si256(last_element, 14); // Shift left by 14 bytes (16-bit element)

    // Shift the rest of the elements right by one position
    __m256i shifted_vec = _mm256_slli_si256(vec, 2); // Shift right by 2 bytes (16-bit element)

    // Combine the shifted elements with the last element
    __m256i result = _mm256_or_si256(shifted_vec, last_element);


    result = _mm256_and_si256(result, result);

    return result;
}

__m256i rotate_right_one(__m256i vec) {
    // Shift the last element to the first position
    __m256i last_element = _mm256_permute2x128_si256(vec, vec, 0x21); // Swap the 128-bit lanes
    last_element = _mm256_srli_si256(last_element, 14); // Shift left by 14 bytes (16-bit element)

    // Shift the rest of the elements right by one position
    __m256i shifted_vec = _mm256_slli_si256(vec, 2); // Shift right by 2 bytes (16-bit element)

    // Combine the shifted elements with the last element
    __m256i result = _mm256_or_si256(shifted_vec, last_element);

    return result;
}

__m256i rotate_left_one(__m256i vec) {
    // Shift the first element to the last position
    __m256i first_element = _mm256_permute2x128_si256(vec, vec, 0x21); // Swap the 128-bit lanes
    first_element = _mm256_slli_si256(first_element, 14); // Shift right by 14 bytes (16-bit element)

    // Shift the rest of the elements left by one position
    __m256i shifted_vec = _mm256_srli_si256(vec, 2); // Shift left by 2 bytes (16-bit element)

    // Combine the shifted elements with the first element
    __m256i result = _mm256_or_si256(shifted_vec, first_element);

    return result;
}

__m256i rotate_right(__m256i vec, int n) {
    // Ensure n is within valid bounds by taking mod 16
    n = ((n % 16) + 32) % 16;

    if (n > 8){
        vec = _mm256_permute2x128_si256(vec, vec, 1);
        n -= 8;
    }

    __m256i last_element;
    __m256i shifted_vec;

    switch (n) {
        case 1:
            last_element = _mm256_permute2x128_si256(vec, vec, 1);
            last_element = _mm256_srli_si256(last_element, 16 - 2);
            shifted_vec = _mm256_slli_si256(vec, 2);
            break;
        case 2:
            last_element = _mm256_permute2x128_si256(vec, vec, 1);
            last_element = _mm256_srli_si256(last_element, 16 - 4);
            shifted_vec = _mm256_slli_si256(vec, 4);
            break;
        case 3:
            last_element = _mm256_permute2x128_si256(vec, vec, 1);
            last_element = _mm256_srli_si256(last_element, 16 - 6);
            shifted_vec = _mm256_slli_si256(vec, 6);
            break;
        case 4:
            last_element = _mm256_permute2x128_si256(vec, vec, 1);
            last_element = _mm256_srli_si256(last_element, 16 - 8);
            shifted_vec = _mm256_slli_si256(vec, 8);
            break;
        case 5:
            last_element = _mm256_permute2x128_si256(vec, vec, 1);
            last_element = _mm256_srli_si256(last_element, 16 - 10);
            shifted_vec = _mm256_slli_si256(vec, 10);
            break;
        case 6:
            last_element = _mm256_permute2x128_si256(vec, vec, 1);
            last_element = _mm256_srli_si256(last_element, 16 - 12);
            shifted_vec = _mm256_slli_si256(vec, 12);
            break;
        case 7:
            last_element = _mm256_permute2x128_si256(vec, vec, 1);
            last_element = _mm256_srli_si256(last_element, 16 - 14);
            shifted_vec = _mm256_slli_si256(vec, 14);
            break;
        case 8:
            last_element = _mm256_permute2x128_si256(vec, vec, 1);
            last_element = _mm256_srli_si256(last_element, 16 - 16);
            shifted_vec = _mm256_slli_si256(vec, 16);
            break;
        default:
            last_element = vec;
            shifted_vec = vec;
            break;
    }

    __m256i result = _mm256_or_si256(shifted_vec, last_element);
    return result;
}

void print_m256i_as_int16(__m256i vec) {
    short values[16];
    _mm256_storeu_si256((__m256i*)values, vec);
    for (int i = 0; i < 16; i++) {
        printf("%hu ", values[i]);
    }
    printf("\n");
}

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

__m256i log2_epi16(__m256i input) {
    __m256i result = _mm256_set1_epi16(16);
    __m256i zero = _mm256_setzero_si256();
    __m256i one = _mm256_set1_epi16(1);

    for (int i = 0; i < 16; i++) {
        __m256i cmp = _mm256_and_si256(_mm256_cmpeq_epi16(input, zero), one);
        result = _mm256_subs_epu16(result, cmp);
        input = _mm256_srli_epi16(input, 1);
    }

    return result;
}

__m256i btc_epi16(__m256i input) {
    __m256i result = _mm256_setzero_si256();
    __m256i zero = _mm256_setzero_si256();
    __m256i one = _mm256_set1_epi16(1);

    for (int i = 0; i < 16; i++) {
        result = _mm256_adds_epu16(result, _mm256_and_si256(input, one));
        input = _mm256_srli_epi16(input, 1);
    }

    return result;
}

__m256i cut_epi16(__m256i input, int n) {
    __m256i mask = _mm256_setr_epi16((16 - n)>0?0xFFFF:0, (16 - n)>1?0xFFFF:0, (16 - n)>2?0xFFFF:0, (16 - n)>3?0xFFFF:0,
                                    (16 - n)>4?0xFFFF:0, (16 - n)>5?0xFFFF:0, (16 - n)>6?0xFFFF:0, (16 - n)>7?0xFFFF:0,
                                    (16 - n)>8?0xFFFF:0, (16 - n)>9?0xFFFF:0, (16 - n)>10?0xFFFF:0, (16 - n)>11?0xFFFF:0,
                                    (16 - n)>12?0xFFFF:0, (16 - n)>13?0xFFFF:0, (16 - n)>14?0xFFFF:0, (16 - n)>15?0xFFFF:0);
    __m256i result = _mm256_and_si256(input, mask);
    return result;
}

__m256i add_all_epi16(__m256i input) {
    __m256i inverted = _mm256_permute2x128_si256(input, input, 0x21);
    input = _mm256_hadd_epi16(input, inverted);
    input = _mm256_hadd_epi16(input, ZERO);
    input = _mm256_hadd_epi16(input, ZERO);
    input = _mm256_hadd_epi16(input, ZERO);
    __m256i result = cut_epi16(input, 15);
    return result;
}

// __m256i max_all_epi16(__m256i input) {
//     __m256i inverted = _mm256_permute2x128_si256(input, input, 0x21);
//     __m256i sums = _mm256_hadd_epi16(input, inverted);
//     __m256i subs = _mm256_hsub_epi16(input, inverted);
//     input = _mm256_abs_epi16(subs);
//     input = _mm256_add_epi16(sums, input);
//     sums = _mm256_hadd_epi16(input, inverted);
//     subs = _mm256_hsub_epi16(input, inverted);
//     input = _mm256_abs_epi16(subs);
//     input = _mm256_add_epi16(sums, input);
//     sums = _mm256_hadd_epi16(input, inverted);
//     subs = _mm256_hsub_epi16(input, inverted);
//     input = _mm256_abs_epi16(subs);
//     input = _mm256_add_epi16(sums, input);
//     sums = _mm256_hadd_epi16(input, inverted);
//     subs = _mm256_hsub_epi16(input, inverted);
//     input = _mm256_abs_epi16(subs);
//     input = _mm256_add_epi16(sums, input);
//     __m256i result = cut_epi16(input, 15);
//     result = _mm256_srli_epi16(result, 4);
//     return result;
// }

__m256i max_all_epi16(__m256i input) {
    __m256i inverted = _mm256_permute2x128_si256(input, input, 0x21);
    input = _mm256_max_epi16(input, inverted);
    input = _mm256_max_epi16(input, _mm256_permute4x64_epi64(input, _MM_SHUFFLE(2, 3, 0, 1)));
    input = _mm256_max_epi16(input, _mm256_permute4x64_epi64(input, _MM_SHUFFLE(1, 0, 3, 2)));
    input = _mm256_max_epi16(input, _mm256_srli_si256(input, 2));
    input = _mm256_max_epi16(input, _mm256_srli_si256(input, 4));
    input = _mm256_max_epi16(input, _mm256_srli_si256(input, 8));
    __m256i result = cut_epi16(input, 15);
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
    __m256i three = _mm256_set1_epi16(3);

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
    __m256i max_height = max_all_epi16(heights);
    __m256i max_holes = max_all_epi16(holes);
    __m256i height_diffs = cut_epi16(_mm256_abs_epi16(_mm256_sub_epi16(rotated_left, heights)), 16-9);
    // __m256i sum_diffs = add_all_epi16(height_diffs);
    __m256i max_diff = max_all_epi16(height_diffs);
    __m256i max_cav = max_all_epi16(cavities);
    __m256i wells_deepness = cut_epi16(_mm256_min_epu16(_mm256_subs_epu16(rotated_right, heights), _mm256_subs_epu16(rotated_left, heights)), 16-10);
    __m256i wells_mask = _mm256_cmpgt_epi16(wells_deepness, three);
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
    for (int i = 0; i < 3; i++){
        t[10*i] = true;
    }
    for (int i = 0; i < 3; i++){
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

    return 0;
}