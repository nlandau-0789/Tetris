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
    // step 1
    __m256i inverted = _mm256_permute2x128_si256(input, input, 0x21);
    input = _mm256_permute2x128_si256(_mm256_max_epi16(input, inverted), _mm256_min_epi16(input, inverted), 0x30);
    print_m256i_as_int16(input);

    // step 2
    inverted = _mm256_permute4x64_epi64(input, _MM_SHUFFLE(1, 0, 3, 2));
    input = _mm256_blend_epi32(
        _mm256_max_epi16(input, inverted), 
        _mm256_min_epi16(input, inverted), 
        0xcc
    );
    print_m256i_as_int16(input);
    
    // step 3
    inverted = _mm256_shufflelo_epi16(input, _MM_SHUFFLE(2, 3, 0, 1));
    inverted = _mm256_shufflehi_epi16(inverted, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epi16(input, inverted),
        _mm256_min_epi16(input, inverted),
        0xcc
    );
    print_m256i_as_int16(input);
    
    // step 4
    inverted = _mm256_shufflehi_epi16(input, _MM_SHUFFLE(1, 0, 3, 2));
    inverted = _mm256_shufflelo_epi16(inverted, _MM_SHUFFLE(1, 0, 3, 2));
    input = _mm256_blend_epi16(
        _mm256_max_epi16(input, inverted),
        _mm256_min_epi16(input, inverted),
        0xaa
    );
    print_m256i_as_int16(input);
    return input;
}

// test the bitonic sort function
int main() {
    __m256i input = _mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    __m256i input2 = _mm256_setr_epi16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    __m256i input3 = _mm256_setr_epi16(8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i input4 = _mm256_setr_epi16(0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8);
    __m256i sorted = bitonic_sort_epu16(input);
    printf("\n");
    __m256i sorted2 = bitonic_sort_epu16(input2);
    printf("\n");
    __m256i sorted3 = bitonic_sort_epu16(input3);
    printf("\n");
    __m256i sorted4 = bitonic_sort_epu16(input4);
    printf("\n");
    
    return 0;
}