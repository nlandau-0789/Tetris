#define TEST_MODEL
#include "v5.c"

int main(){
    init_piece_placements();
    
    nn *network = malloc(sizeof(nn));
    // load_nn(network, "networks/2504855.model"); // 844k
    load_nn(network, "networks/3912987.model"); 
    unsigned long long score = 0, max_score = 0, ttime = 0;
    int n_games = 10000;
    int seed = time(NULL);
    for (int i = 0; i < n_games; i++){
        srand(seed);
        int start_time = time(NULL);
        int next_seed = rand();
        int game_score = play_full_game(network, seed);
        score += game_score;
        if (game_score > max_score) max_score = game_score;
        seed = next_seed;
        int end_time = time(NULL);
        ttime += (end_time - start_time);
        printf("%d : %lld %lld %lld/s     \r", (i+1), score/(i+1), max_score, score * 5 / 2 /ttime);
        fflush(stdout);
    }
}