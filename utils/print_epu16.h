#ifndef IMPORT_STD_H
#include "std.h"
#define IMPORT_STD_H
#endif


void print_m256i_as_int16(__m256i vec) {
    short values[16];
    _mm256_storeu_si256((__m256i*)values, vec);
    for (int i = 0; i < 16; i++) {
        printf("%hu ", values[i]);
    }
    printf("\n");
}

#define PRINT_EPU16_H