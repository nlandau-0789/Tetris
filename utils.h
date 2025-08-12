#ifndef UTILS_H
#define UTILS_H
#include <immintrin.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

void print_m256i_as_int16(__m256i vec) {
    short values[16];
    _mm256_storeu_si256((__m256i*)values, vec);
    for (int i = 0; i < 16; i++) {
        printf("%hu ", values[i]);
    }
    printf("\n");
}

__m256i bitonic_sort_epu16(__m256i input) {
    // print_m256i_as_int16(input);
    __m256i inverted;

    // 2
    // step 1
    inverted = _mm256_shufflehi_epi16(input, _MM_SHUFFLE(2, 3, 0, 1));
    inverted = _mm256_shufflelo_epi16(inverted, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x66
    );
    // print_m256i_as_int16(input);

    // 4
    // step 2
    inverted = _mm256_shuffle_epi32(input, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x3c
    );
    // print_m256i_as_int16(input);

    // step 3
    inverted = _mm256_shufflehi_epi16(input, _MM_SHUFFLE(2, 3, 0, 1));
    inverted = _mm256_shufflelo_epi16(inverted, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x5a
    );
    // print_m256i_as_int16(input);

    // 8
    // step 4
    inverted = _mm256_permute4x64_epi64(input, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi32(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x3c
    );
    // print_m256i_as_int16(input);

    // step 5
    inverted = _mm256_shuffle_epi32(input, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi32(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x5a
    );
    // print_m256i_as_int16(input);

    // step 6
    inverted = _mm256_shufflehi_epi16(input, _MM_SHUFFLE(2, 3, 0, 1));
    inverted = _mm256_shufflelo_epi16(inverted, _MM_SHUFFLE(2, 3, 0, 1));
    __m256i minvec = _mm256_min_epu16(input, inverted);
    __m256i maxvec = _mm256_max_epu16(input, inverted);
    input = _mm256_permute2x128_si256(
        _mm256_blend_epi16(
            maxvec,
            minvec,
            0x55
        ), 
        _mm256_blend_epi16(
            maxvec,
            minvec,
            0xaa
        ), 
        0x12
    );
    // print_m256i_as_int16(input);
    
    // 16
    // step 7
    inverted = _mm256_permute2x128_si256(input, input, 0x21);
    input = _mm256_permute2x128_si256(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x21
    );
    // print_m256i_as_int16(input);

    // step 8
    inverted = _mm256_permute4x64_epi64(input, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi32(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0xcc
    );
    // print_m256i_as_int16(input);

    // step 9
    inverted = _mm256_shuffle_epi32(input, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0xcc
    );
    // print_m256i_as_int16(input);

    // step 10
    inverted = _mm256_shufflehi_epi16(input, _MM_SHUFFLE(2, 3, 0, 1));
    inverted = _mm256_shufflelo_epi16(inverted, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0xaa
    );
    // print_m256i_as_int16(input);


    return input;
}

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
    // __m256i zero = _mm256_setzero_si256();
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

int is_zero_m256i(__m256i vec) {
    // Compare each element in vec to zero; this will set each element in the result to all 1s (if equal) or all 0s (if not equal)
    __m256i cmp_result = _mm256_cmpeq_epi32(vec, _mm256_setzero_si256());

    // Check if all comparisons resulted in true (i.e., all elements are zero)
    uint32_t mask = _mm256_movemask_epi8(cmp_result);

    // If all elements are zero, the mask will be 0xFFFFFFFF
    return mask == 0xFFFFFFFF;
}

struct _m256i_vector {
    __m256i *data;
    size_t size;
    size_t capacity;
};

typedef struct _m256i_vector _m256i_vector;

_m256i_vector init_mm256i_vector(size_t capacity) {
    _m256i_vector vec;
    vec.data = (__m256i *)malloc(capacity * sizeof(__m256i));
    vec.size = 0;
    vec.capacity = capacity;
    return vec;
}

void free_mm256i_vector(_m256i_vector *vec) {
    free(vec->data);
    vec->size = 0;
    vec->capacity = 0;
    vec->data = NULL;
}

void append_mm256i_vector(_m256i_vector *vec, __m256i value) {
    if (vec->size >= vec->capacity) {
        vec->capacity *= 2;
        vec->data = (__m256i *)realloc(vec->data, vec->capacity * sizeof(__m256i));
    }
    vec->data[vec->size] = value;
    vec->size++;
}

__m256i pop_m256i_vector(_m256i_vector *vec) {
    if (vec->size == 0) {
        fprintf(stderr, "Error: Attempt to pop from an empty vector.\n");
        exit(EXIT_FAILURE);
    }
    vec->size--;
    return vec->data[vec->size];
}

#include "rng.c"
#include "pool.c"

#endif