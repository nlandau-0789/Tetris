// make the main function such that it makes 2 _m256i vectors, and showcases the use of _subs_epu16

#include <immintrin.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

int main() {
    __m256i input = _mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    __m256i inverted = _mm256_set_epi16(8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i result = _mm256_subs_epu16(input, inverted);
    
    // Print the result
    int16_t res[16];
    _mm256_storeu_si256((__m256i*)res, result);
    for (int i = 0; i < 16; i++) {
        printf("%d ", res[i]);
    }
    printf("\n");
    
    return 0;
}