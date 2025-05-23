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

    // 2
    // step 1
    inverted = _mm256_shufflehi_epi16(input, _MM_SHUFFLE(2, 3, 0, 1));
    inverted = _mm256_shufflelo_epi16(inverted, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x66
    );
    print_m256i_as_int16(input);

    // 4
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

    // 8
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
    input = _mm256_blend_epi32(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x5a
    );
    print_m256i_as_int16(input);

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
    print_m256i_as_int16(input);
    
    // 16
    // step 7
    inverted = _mm256_permute2x128_si256(input, input, 0x21);
    input = _mm256_permute2x128_si256(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0x21
    );
    print_m256i_as_int16(input);

    // step 8
    inverted = _mm256_permute4x64_epi64(input, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi32(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0xcc
    );
    print_m256i_as_int16(input);

    // step 9
    inverted = _mm256_shuffle_epi32(input, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0xcc
    );
    print_m256i_as_int16(input);

    // step 10
    inverted = _mm256_shufflehi_epi16(input, _MM_SHUFFLE(2, 3, 0, 1));
    inverted = _mm256_shufflelo_epi16(inverted, _MM_SHUFFLE(2, 3, 0, 1));
    input = _mm256_blend_epi16(
        _mm256_max_epu16(input, inverted),
        _mm256_min_epu16(input, inverted),
        0xaa
    );
    print_m256i_as_int16(input);


    return input;
}

// test the bitonic sort function
int main() {
    __m256i input = _mm256_setr_epi16(6, 7, 1, 5, 7, 10, 6, 3, 8, 8, 3, 8, 3, 8, 7, 2);
    __m256i sorted = bitonic_sort_epu16(input);
    return 0;
}