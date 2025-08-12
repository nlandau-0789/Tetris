#include "rng.c"
#include <stdlib.h>

struct pool {
    void **samples;
    int capacity, size, count;
    int *rand_seed;
    void (*free_t)(void *);
};

typedef struct pool pool;

pool* init_pool(int capacity, void (*free_t)(void *), int seed){
    pool *p = malloc(sizeof(pool));
    p->samples = malloc(sizeof(void *) * capacity);
    p->capacity = capacity;
    p->size = 0;
    p->count = 0;
    p->rand_seed = malloc(sizeof(int));
    *(p->rand_seed) = seed;
    p->free_t = free_t;
    return p;
}

void free_pool(pool *p){
    free(p->rand_seed);
    for (int i = 0; i < p->size; i++){
        p->free_t(p->samples[i]);
    }
    free(p->samples);
    free(p);
}

void add_to_pool(pool *p, void *sample){
    p->count++;
    if (p->size < p->capacity){
        p->samples[p->size] = sample;
        p->size++;
        return;
    }
    int i = draw(p->rand_seed, 0, p->count);
    if (i < p->capacity){
        p->free_t(p->samples[i]);
        p->samples[i] = sample;
    } else {
        p->free_t(sample);
    }
}

#ifdef TEST_POOLS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(){
    srand(time(NULL));
    for (int j = 0; j < 1000000; j++){
        pool *p = init_pool(3, free, j);
        
        for (int i = 0; i < 10; i++){
            int *j = malloc(sizeof(int));
            *j = i;
            add_to_pool(p, j);
        }


        for (int i = 0; i < p->capacity; i++){
            printf("%d ", *(int *)(p->samples[i]));
        }
        printf("\n");
        free_pool(p);
    }
}
#endif