#ifdef EVO_TRAIN_C
void train_nn(nn *generation[], int gen_size, int n_games, int n_gen, int n_hidden_layers, int hidden_layer_sizes[], int start_gen) {
    nn **new_generation = malloc(gen_size * sizeof(nn*));

    nn start;
    if (start_gen > 0){
        char buffer[100];
        sprintf(buffer, "./games/gen%d", start_gen);
        load_nn(&start, buffer);
    }
    // reward_t rewards[] = {phase2_rew};
    for (int i = 0; i < gen_size; i++) {
        generation[i] = malloc(sizeof(nn));
        init_nn(generation[i], NN_INPUT_SIZE, n_hidden_layers, hidden_layer_sizes, 40, 1.0f, time(NULL)+i);
        if (start_gen > 0){
            weight_avg_nn(generation[i], &start, (float)i/gen_size);
        }
        // rl_train(generation[i], 6000, 1, 0.0001f, 0.1f, 0.25f, rewards, 1);
        // printf("pretrained %d/%d\r", i+1, gen_size);
        // fflush(stdout);
    }
    for (int gen = start_gen + 1; gen < n_gen; gen++) {
        int *scores = malloc(gen_size * sizeof(int));
        int *seeds = malloc(n_games * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < n_games; i++) {
            seeds[i] = rand();
        }
        for (int i = 0; i < gen_size; i++) {
            scores[i] = 0;
            for (int j = 0; j < n_games; j++) {
                int turns = play_full_game(generation[i], seeds[j]);
                scores[i] += turns;
                printf("            \r[%d] Game %02d/%d of instance %04d/%d, score: %d", gen, j+1, n_games, i+1, gen_size, turns);
                // fflush(stdout);
            }
            // printf("%d ", scores[i]);
        }

        for (int i = 0; i < gen_size - 1; i++) {
            for (int j = i + 1; j < gen_size; j++) {
                if (scores[i] < scores[j]) {
                    nn *temp = generation[i];
                    generation[i] = generation[j];
                    generation[j] = temp;
                    
                    int temp_score = scores[i];
                    scores[i] = scores[j];
                    scores[j] = temp_score;
                }
            }
        }
        
        // printf("\n");
        // for (int i = 0; i < 10; i++) {
        //     printf("%d ", scores[i]);
        // }
        // printf("\n");

        char buffer[100];
        sprintf(buffer, "./games/gen%d", gen);
        save_nn(generation[0], buffer);
        // FILE * f = fopen(buffer, "w");
        // print_full_game(f, generation[0], time(NULL));
        // fclose(f);

        for (int i = 0; i < gen_size; i++) {
            new_generation[i] = malloc(sizeof(nn));
            init_nn(new_generation[i], NN_INPUT_SIZE, n_hidden_layers, hidden_layer_sizes, 1.0f, time(NULL)+i);
            weight_avg_nn(new_generation[i], generation[i/20], (float)i/gen_size);
            // printf("%d ", scores[i]);
        }
        for (int i = 0; i < gen_size; i++) {
            free_nn(generation[i]);
            free(generation[i]);
            generation[i] = new_generation[i];
        }
        printf("[%d] best : %d\n", gen, scores[0]);
        free(scores);
    }
    free(new_generation);

}
#endif