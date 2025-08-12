#include <stdlib.h>

int draw(int *rng, int start, int end){
    srand(*rng);
    *rng = rand();
    return (*rng) % (end-start) + start;
}

float draw_f(int *rng, float start, float end){
    srand(*rng);
    *rng = rand();
    return ((float)(*rng) / RAND_MAX) * (end-start)+start;
}

double draw_d(int *rng, double start, double end){
    srand(*rng);
    *rng = rand();
    return ((double)(*rng) / RAND_MAX) * (end-start)+start;
}