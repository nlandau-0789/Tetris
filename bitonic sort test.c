#include <immintrin.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

void print_m256i_as_int16(__m256i vec) {
    short values[16];
    _mm256_storeu_si256((__m256i*)values, vec);
    for (int i = 0; i < 16; i++) {
        printf("%hu ", values[i]);
    }
    printf("\n");
}

__m256i bitonic_sort_epu16(__m256i input) {
    print_m256i_as_int16(input);
    __m256i inverted;


    // step 1
    inverted = _mm256_shufflehi_epi16(input, _MM_SHUFFLE(2, 3, 0, 1));
    inverted = _mm256_shufflelo_epi16(inverted, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x66
    );
    print_m256i_as_int16(input);
    

    // step 2
    inverted = _mm256_shuffle_epi32(input, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x3c
    );
    print_m256i_as_int16(input);

    // step 3
    inverted = _mm256_shufflehi_epi16(input, _MM_SHUFFLE(2, 3, 0, 1));
    inverted = _mm256_shufflelo_epi16(inverted, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x5a
    );
    print_m256i_as_int16(input);

    // step 4
    inverted = _mm256_permute4x64_epi64(input, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi32(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x3c
    );
    print_m256i_as_int16(input);

    // step 5
    inverted = _mm256_shuffle_epi32(input, _MM_SHUFFLE(2, 3, 0, 1));
    print_m256i_as_int16(inverted);
    input = _mm256_blend_epi32(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x5a
    );
    print_m256i_as_int16(input);

    return input;
}

// test the bitonic sort function
int main() {
    __m256i input = _mm256_setr_epi16(13, 10, 8, 1, 10, 4, 3, 15, 8, 10, 2, 7, 7, 5, 16, 15);
    __m256i sorted = bitonic_sort_epu16(input);
    return 0;
}