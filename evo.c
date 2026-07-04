#include "v5.c"

#define CONSTANTE_UCB 3

struct creature {
    __m256i *weights;
    int n_played;
    double total_score;
};

typedef struct creature creature;

creature* init_creature(){
    creature* c = malloc(sizeof(creature));
    creature template = {malloc(sizeof(__m256i) * N_FEATURES), 0, 0.0};
    *c = template; 
    for (int i = 0; i < N_FEATURES; i++){
        c->weights[i] = _mm256_set_epi16(0, 0, 0, 0, 0, 0, randint(INT16_MIN, INT16_MAX), randint(INT16_MIN, INT16_MAX), randint(INT16_MIN, INT16_MAX), randint(INT16_MIN, INT16_MAX), randint(INT16_MIN, INT16_MAX), randint(INT16_MIN, INT16_MAX), randint(INT16_MIN, INT16_MAX), randint(INT16_MIN, INT16_MAX), randint(INT16_MIN, INT16_MAX), randint(INT16_MIN, INT16_MAX));
    }
    return c;
}

void free_creature(creature *c){
    free(c->weights);
    free(c);
}

void save_creature(creature *c, char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) return;

    int16_t temp[16];
    for (int i = 0; i < N_FEATURES; i++) {
        _mm256_storeu_si256((__m256i*)temp, c->weights[i]);
        
        for (int j = 0; j < 10; j++) {
            fprintf(f, "%d ", temp[j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

creature *load_creature(char *filename) {
    FILE *f = fopen(filename, "w");
    creature *c = init_creature();
    if (!f) return c;
    int16_t temp[16] = {0};
    for (int i = 0; i < N_FEATURES; i++) {
        for (int j = 0; j < 10; j++) {
            fscanf(f, "%hd ", &temp[j]);
        }
        c->weights[i] = _mm256_set_epi16(0, 0, 0, 0, 0, 0, temp[9], temp[8], temp[7], temp[6], temp[5], temp[4], temp[3], temp[2], temp[1], temp[0]);
    }
    fclose(f);
    return c;
}

// segtree pour trouver le max UCB score
struct segtree {
    creature **creatures;
    double *ucb_scores;
    int size, n_samples;
};

typedef struct segtree segtree;

segtree* init_segtree(int population){
    int size = population * 2;
    segtree *t = malloc(sizeof(segtree));
    segtree template = {malloc(sizeof(creature*) * size),
                        malloc(sizeof(double) * size),
                        size, 0};
    *t = template;
    for (int i = population; i < size; i++){
        t->creatures[i] = init_creature();
        t->ucb_scores[i] = INFINITY;
    }
    for (int i = population - 1; i > 0; i--){
        t->creatures[i] = t->creatures[i*2];
        t->ucb_scores[i] = t->ucb_scores[i*2];
    }
    return t;
}

void free_segtree(segtree *t){
    for (int i = t->size/2; i < t->size; i++){
        free_creature(t->creatures[i]);
    }
    free(t->creatures);
    free(t->ucb_scores);
    free(t);
}

double ucb_score(creature *c, int t){
    return (c->total_score/c->n_played) + CONSTANTE_UCB * sqrt(log((double)(t + 1)) / c->n_played);
}

void update_up_segtree(segtree *t, int idx, double new_score){ // idx doit etre > t->size/2 
    t->ucb_scores[idx] = new_score;
    while (idx > 1){
        if (t->ucb_scores[idx] > t->ucb_scores[idx ^ 1]){
            t->ucb_scores[idx/2] = t->ucb_scores[idx];
            t->creatures[idx/2] = t->creatures[idx];
        } else {
            t->ucb_scores[idx/2] = t->ucb_scores[idx ^ 1];
            t->creatures[idx/2] = t->creatures[idx ^ 1];
        }
        idx /= 2;
    }
}

void update_down_segtree(segtree *t){
    int idx = 1;
    double score = ucb_score(t->creatures[idx], t->n_samples); 
    while (idx < t->size / 2){
        int next_idx;
        if (t->creatures[idx*2] == t->creatures[idx]){
            next_idx = idx * 2;
        } else {
            next_idx = idx * 2 + 1;
        }
        idx = next_idx;
    }
    update_up_segtree(t, idx, score);
}

void verifier_segtree(segtree *t){
    for (int i = 1; i < t->size / 2; i++){
        int child;
        if (t->creatures[i] == t->creatures[i*2]){
            child = i * 2;
        } else if (t->creatures[i] == t->creatures[i*2+1]) {
            child = i * 2 + 1;
        } else {
            printf("pb1\n");
            return;
        }
        if (t->ucb_scores[child] < t->ucb_scores[child ^ 1]){
            printf("pb2\n");
        }
    }
}

creature *max_segtree(segtree *t){
    return t->creatures[1];
}

int ask_creature(creature *c, __m256i board){
    calc_consts(board);
    __m256i sum = ZERO; 
    for (int i = 0; i < N_FEATURES; i++){
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(consts[i], c->weights[i]));
    }
    int total = reduce_add_all_epi32(sum);
    return total;
}

placement get_placement(int piece, __m256i board, creature *c) {
    int best_eval = INT_MIN;
    placement best_placement = {0, 0, 0, true};
    for (int r = 0; r < n_rot[piece]; r++) {
        __m256i piece_board_init = placements[piece][r];
        for (int x = 0; x < 10-piece_width[piece][r]+1; x++) { // potential +/-1 mistake
            #ifdef DEBUG_VERBOSE
            // printf("Trying piece %d, rotation %d, x=%d\n", piece, r, x);
            #endif
            int y = 16;
            __m256i piece_board = piece_board_init;
            if (!is_zero_m256i(_mm256_and_si256(piece_board, board))){
                #ifdef DEBUG_VERBOSE
                fprint_board(log_file, board);
                fprint_board(log_file, piece_board);
                fprintf(log_file, "Piece overlaps with board at x=%d, r=%d\n", x, r);
                #endif
                piece_board_init = rotate_right_one(piece_board_init);
                continue; // piece overlaps with board
            }
            while (y > piece_height[piece][r] && is_zero_m256i(_mm256_and_si256(_mm256_srli_epi16(piece_board, 1), board))){ // potential +/-1 mistake
                y--;
                piece_board = _mm256_srli_epi16(piece_board, 1);
                #ifdef DEBUG_VERBOSE
                fprint_board(log_file, piece_board);
                #endif
            }

            // evaluate the placement
            __m256i new_board = _mm256_or_si256(board, piece_board);
            remove_full_lines(&new_board);

            int eval = ask_creature(c, new_board);

            if (eval > best_eval) {
                best_eval = eval;
                best_placement.x = x;
                best_placement.r = r;
                best_placement.eval = eval;
                best_placement.dead = false;
            }
            
            piece_board_init = rotate_right_one(piece_board_init);
        }
    }
    return best_placement;
}

double score(int turn){
    // return (log((double)turn) - 3.0)/9;
    return (double)turn - 20;
}

void sample(creature *c){
    __m256i board = ZERO;
    int piece = randint(0, 7);
    int n_lines_removed = 0;

    for (int turn = 0; turn < 2000000000; turn++) {
        placement placement = get_placement(piece, board, c);
        #ifdef DEBUG_VERBOSE
        fflush(stdout);
        printf("%f %d %d\n", placement.eval, placement.x, placement.r);
        #endif
        if (placement.dead) {
            // printf("Game over! Dead at turn %d\n", turn);
            c->n_played++;
            c->total_score += score(turn);
            return;
        }
        board = place(piece, placement.x, placement.r, board, &n_lines_removed);
        #ifdef DEBUG_VERBOSE
        print_board(board);
        #endif
        piece = randint(0, 7);
    }
    c->n_played++;
    c->total_score += score(2000000000);
}

void print_creature(creature *c) {
    short temp[16];
    for (int i = 0; i < N_FEATURES; i++) {
        _mm256_storeu_si256((__m256i*)(&temp), c->weights[i]);
        for (int j = 0; j < 10; j++) {
            printf("%hd ", temp[j]);
        }
        printf("\n");
    }
}

void print_segtree(segtree *t){
    for (int i = 1; i < t->size; i++){
        printf("%lf ", t->ucb_scores[i]);
        if (!(i & (i+1))) {
            printf("\n");
        }
    }
    printf("\n");
    for (int i = 1; i < t->size; i++){
        printf("%d ", t->creatures[i]->n_played);
        if (!(i & (i+1))) {
            printf("\n");
        }
    }
    printf("\n");
    for (int i = 1; i < t->size; i++){
        printf("%lf ", ((t->creatures[i]->total_score)/(t->creatures[i]->n_played)));
        if (!(i & (i+1))) {
            printf("\n");
        }
    }
    printf("\n");
    printf("\n");
}

void evaluate_gen(segtree *gen, int n_rollouts){
    for (int i = 0; i < n_rollouts; i++){
        creature *c = max_segtree(gen);
        sample(c);
        gen->n_samples++;
        #ifdef DEBUG_VERBOSE
        print_segtree(gen);
        #endif
        update_down_segtree(gen);
        #ifdef DEBUG_VERBOSE
        verifier_segtree(gen);
        print_segtree(gen);
        #endif
    }
}

void train(segtree *gen, int nb_rollouts, int nb_gen){
    for (int gen_idx = 0; gen_idx < nb_gen; gen_idx++){
        evaluate_gen(gen, nb_rollouts);
        char filename[200];
        sprintf(filename, "./heuristics/%d", (int)(gen->creatures[1]->total_score / gen->creatures[1]->n_played));
        save_creature(gen->creatures[1], filename);
        for (int i = 2; i < gen->size; i++){
            int step_size = i * i / 2;
            if (gen->creatures[i] == gen->creatures[i/2]){
                gen->ucb_scores[i] = gen->ucb_scores[i/2];
                continue;
            }
            gen->n_samples -= gen->creatures[i]->n_played;
            gen->ucb_scores[i] = INFINITY;
            gen->creatures[i]->n_played = 0;
            gen->creatures[i]->total_score = 0;
            for (int j = 0; j < N_FEATURES; j++){
                gen->creatures[i]->weights[j] = _mm256_add_epi16(gen->creatures[i / 2]->weights[j], _mm256_set_epi16(0, 0, 0, 0, 0, 0, randint(-step_size, step_size), randint(-step_size, step_size), randint(-step_size, step_size), randint(-step_size, step_size), randint(-step_size, step_size), randint(-step_size, step_size), randint(-step_size, step_size), randint(-step_size, step_size), randint(-step_size, step_size), randint(-step_size, step_size)));
            }
        }
        printf("%lf\n", gen->creatures[1]->total_score / gen->creatures[1]->n_played);
        // print_segtree(gen);
        update_down_segtree(gen);
        fflush(stdout);
    }
}

int main(){
    init_piece_placements();
    set_random_seed(time(NULL));
    segtree *gen = init_segtree(1<<8);
    // evaluate_gen(gen, 10000);
    train(gen, 1<<10, 10000);
    free_segtree(gen);
}