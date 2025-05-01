#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

void print_m256i_as_int16(__m256i vec) {
    short values[16];
    _mm256_storeu_si256((__m256i*)values, vec);
    for (int i = 0; i < 16; i++) {
        printf("%hu ", values[i]);
    }
    printf("\n");
}

// Function to approximate the integer part of log2 of each 16-bit integer in a __m256i vector
__m256i log2_epi16(__m256i input) {
    // Step 1: Find the MSB position
    // Use a bit scan reverse (BSR) technique to find the position of the MSB
    __m256i result = _mm256_set1_epi16(16);
    __m256i zero = _mm256_setzero_si256();
    __m256i one = _mm256_set1_epi16(1);

    for (int i = 0; i < 16; i++) {
        print_m256i_as_int16(input); // Debugging line to see the intermediate values
        __m256i cmp = _mm256_and_si256(_mm256_cmpeq_epi16(input, zero), one);
        result = _mm256_subs_epu16(result, cmp);
        input = _mm256_srli_epi16(input, 1);
    }

    return result;
}

int main() {
    // Example usage
    __m256i input = _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 0xabcd, 2048, 4096, 8192, 16384, 0xFFFF);
    __m256i result = log2_epi16(input);

    // Print the result (for demonstration purposes)
    uint16_t output[16];
    _mm256_storeu_si256((__m256i*)output, result);
    for (int i = 0; i < 16; i++) {
        printf("log2(%d) ≈ %d\n", ((uint16_t*)&input)[i], output[i]);
    }

    return 0;
}
