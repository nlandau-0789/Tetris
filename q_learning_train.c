#ifndef NN_INPUT_SIZE
#define NN_INPUT_SIZE 100
#endif

#ifndef MAX_TURNS
#define MAX_TURNS 2000000000
#endif

#ifdef Q_LEARNING_TRAIN_C
float max_reward = -INFINITY;
float min_reward = INFINITY;
float phase2_rew(__m256i old_board, __m256i new_board, int n_lines_removed) {
    float reward = (float)n_lines_removed * 10.0f; // Reward for clearing lines
    if (reward > max_reward) {
        max_reward = reward;
    }
    if (reward < min_reward) {
        min_reward = reward;
    }
    return reward; // Normalize reward to be between 0 and 1
}

float phase1_rew(__m256i old_board, __m256i new_board, int n_lines_removed) {
    // Reward for clearing lines
    // float line_reward = 0.0f;
    // if (n_lines_removed == 1) line_reward = 50.0f;
    // else if (n_lines_removed == 2) line_reward = 90.0f;
    // else if (n_lines_removed == 3) line_reward = 100.0f;
    // else if (n_lines_removed == 4) line_reward = 110.0f;

    // Penalty for holes
    calc_consts(new_board);
    short holes_new = 0;
    for (int i = 0; i < 10; i++) {
        holes_new += ((short*)&consts[2])[i];
    }

    // Penalty for height (use same consts call)
    short max_height = ((short*)&consts[1])[0];
    for (int i = 1; i < 10; i++) {
        if (((short*)&consts[1])[i] > max_height)
            max_height = ((short*)&consts[1])[i];
    }

    float height_penalty = (float)max_height * 2.0f;

    // Now compute holes in old_board
    calc_consts(old_board);
    short holes_old = 0;
    for (int i = 0; i < 10; i++) {
        holes_old += ((short*)&consts[2])[i];
    }

    float hole_penalty = (float)(holes_new - holes_old) * 4.0f;

    // Total reward
    float reward = - hole_penalty - height_penalty;
    reward /= 200.0f;
    if (reward > max_reward) {
        max_reward = reward;
    }
    if (reward < min_reward) {
        min_reward = reward;
    }
    return reward;
}

float phase3_rew(__m256i old_board, __m256i new_board, int n_lines_removed) {
    calc_consts(new_board);
    short max_height = ((short*)&consts[1])[0];
    float reward = 16-max_height;
    return reward / 16.0f;
}

float rew(__m256i old_board, __m256i new_board, int n_lines_removed) {
    if (n_lines_removed > 0) {
        return n_lines_removed * 10.0f; // Reward for clearing lines
    } 
    return 1.0f;
}

typedef float (*reward_t)(__m256i, __m256i, int);

void rl_train(nn *network, int episodes, int seasons, float learning_rate, float gamma_i, float epsilon_i, reward_t reward_calcs[], int n_phases) {
    int log_every = 1;
    for (int s = 0; s < seasons; s++) {
        float epsilon = epsilon_i;
        float gamma = gamma_i;
        int rew_idx = s % n_phases;
        int total_lines_removed = 0;
        double avg_output = 0.0;
        float threshold_lines = 10000;
        for (int ep = 0; ep < episodes; ep++) {
            // Start with an empty board
            __m256i board = ZERO;
            float total_reward = 0;
            int turn = 0;
            int last_piece;
            
            while (turn < MAX_TURNS) {
                int piece = rand() % 7;
                last_piece = piece;
    
                // Epsilon-greedy: random move or best move
                placement best;
                if (((float)rand() / RAND_MAX) < epsilon) {
                    // Random action
                    int r = rand() % n_rot[piece];
                    int x = rand() % (10 - piece_width[piece][r] + 1);
                    // check if the piece can be placed
                    __m256i piece_board = placements[piece][r];
                    best.x = x;
                    best.r = r;
                    best.dead = false;
                    for (int i = 0; i < x; i++) {
                        piece_board = rotate_right_one(piece_board);
                    }
                    if (!is_zero_m256i(_mm256_and_si256(piece_board, board))){
                        best.dead = true; // piece overlaps with board
                    }
                } else {
                    // Greedy action (use NN)
                    best = get_placement(piece, board, network);
                }
    
                // Place the piece
                float reward = 0;
                if (best.dead) {
                    float target = -1.0f;
    
                    float input[NN_INPUT_SIZE];
                    get_nn_input(input, board);
                    for (int epoch = 0; epoch < 3; epoch++) {
                        backpropagate(network, input, target, ReLU, ReLU_derivative, learning_rate / 3.0f);
                    }
                    break;
                }
                int n_lines_removed;
                __m256i new_board = place(piece, best.x, best.r, board, &n_lines_removed);
                reward = reward_calcs[rew_idx](board, new_board, n_lines_removed);
                total_reward += reward;
                total_lines_removed += n_lines_removed;
    
                // Prepare NN input for current state
                float input[NN_INPUT_SIZE], next_input[NN_INPUT_SIZE];
                get_nn_input(input, board);
                get_nn_input(next_input, new_board);
    
                // Q-learning target: reward + gamma * max_a' Q(next_state, a')
                network->input = next_input;
                feed_forward(network, ReLU);
                if (isnan(network->output)) {
                    printf("\n%f, %f\n", max_reward, min_reward);
                    printf("Network output unstable: %f\n", network->output);
                    print_nn(network);
                    exit(1);
                }
                float next_value = (network->output);
                avg_output += next_value;
    
                float target = (reward + gamma * next_value);
                // if (target > 1000.0f) target = 1000.0f;
                // if (target < -1000.0f) target = -1000.0f;
    
                // Train on (state, target)
                backpropagate(network, input, target, ReLU, ReLU_derivative, learning_rate);
    
                // Move to next state
                board = new_board;
                turn++;
                // epsilon *= 0.999f;
            }
            if (turn == 0) {
                get_placement(last_piece, board, network);
            }
            if ((ep + 1) % log_every == 0) {
                // printf("\rEpisode %d finished, avg_lr=%lf, avg_output=%lf        ", ep+1, (double)total_lines_removed/500, avg_output / 500.0);
                printf("%d ", total_lines_removed/log_every);
                avg_output = 0.0;
                fflush(stdout);
                // if (total_lines_removed > threshold_lines * log_every) {
                //     printf("used it");
                //     learning_rate *= 0.1;
                //     threshold_lines *= 10;
                // }
                total_lines_removed = 0;
            } 
            epsilon *= 1.0001;
            if (epsilon > 0.15) {
                epsilon = 0.15f; // maximum epsilon
            }
            // epsilon *= 0.9999;
            // gamma *= 1.0004;
            // if (epsilon < 0.15) {
            //     epsilon = 0.15f; // minimum epsilon
            // }
            // if (gamma > 0.99) {
            //     gamma = 0.99; // maximum gamma
            // }
        }
        // if (total_lines_removed > threshold_lines * 50) {
        //     learning_rate *= 0.9;
        //     threshold_lines *= 10;
        //     epsilon_i *= 0.99;
        //     gamma_i *= 1.01;
        //     if (epsilon_i < 0.1) {
        //         epsilon_i = 0.1; // minimum epsilon
        //     }
        //     if (gamma_i > 0.99) {
        //         gamma_i = 0.99; // maximum gamma
        //     }
        // }
        // printf("\n");
    }
}
#endif
#ifdef BATCHED_TRAIN
struct state {
    float *input;
    float *reward;
}

typedef struct state state;

void free_state(state *s){
    free(s->input);
    free(s->reward);
    free(s);
}

void batched_q_train(nn* network, int n_episodes, int n_batches, int batch_size, float learning_rate, float gamma, float epsilon, float epsilon_decay, reward_t rew) {
    pool *p;
    bool valid_placements[40];
    for (int ep = 0; ep < n_episodes; ep++){
        p = init_pool(n_batches * batch_size, free_state, rand())
        __m256i board = ZERO;
        int turn = 0;
        int n_lines_removed = 0, total_lines_removed = 0;
        while (turn < MAX_TURNS) {
            int piece = rand() % 7;
            int n_valid_placements = get_valid_placements(valid_placements, piece, board);
            state *s = malloc(sizeof(state));
            s->input = malloc(sizeof(float) * NN_INPUT_SIZE);
            s->reward = malloc(sizeof(float) * 40);
            get_nn_input(s->input, board);

            if (n_valid_placements == 0){
                for (int i = 0; i < 40; i++){
                    s->reward[i] = 0; // peut etre mettre -1 ?
                }

                backpropagate(network, s->input, s->reward, ReLU, ReLU_derivative, learning_rate);
                free_state(s);
                break;
            }

            int r, x, a;
            network->input = s->input;
            feed_forward(network, ReLU);
            for (int i = 0; i < 40; i++){
                if (isnan(network->output[i])) {
                    printf("Network output unstable\n");
                    fflush(stdout);
                    exit(1);
                }
                s->reward[i] = 0;
            }

            // Epsilon-greedy: random move or best move
            if (((float)rand() / RAND_MAX) < epsilon) {
                // Random action
                int idx = rand() % n_valid_placements;
                int valid_move_count = 0;
    
                for (int rr = 0; rr < n_rot[piece]; rr++){
                    for (int xx = 0; xx < (10 - piece_width[piece][r] + 1); xx++){
                        if (valid_placements[rr * 10 + xx]){
                            if (valid_move_count == idx){
                                r = rr; 
                                x = xx;
                                a = rr * 10 + xx;
                            }
                            valid_move_count++;
                        }
                    }
                }
            } else {
                // Greedy action (use NN)
                float best_eval = -INFINITY;
                
                for (int rr = 0; rr < n_rot[piece]; rr++){
                    for (int xx = 0; xx < (10 - piece_width[piece][r] + 1); xx++){
                        if (valid_placements[rr * 10 + xx]){
                            if (network->output[rr*10+xx] > best_eval){
                                r = rr; 
                                x = xx;
                                a = rr * 10 + xx;
                                best_eval = network->output[rr*10+xx];
                            }
                        }
                    }
                }
            }

            __m256i new_board = place(piece, x, r, board, &n_lines_removed);
            float reward = rew(board, new_board, n_lines_removed);
            total_lines_removed += n_lines_removed;
            s->reward[a] = reward;

            // float next_input[NN_INPUT_SIZE];
            // get_nn_input(next_input, new_board);

            // network->input = next_input;
            // feed_forward(network, ReLU);
            // float next_value = (network->output);

            // float target = (reward + gamma * next_value);
            // if (target > 1000.0f) target = 1000.0f;
            // if (target < -1000.0f) target = -1000.0f;

            add_to_pool(p, s);

            turn++;
            board = new_board; 
        }
        // printf("\n");
        
        int real_batch_size = p->size < batch_size ? p->size : batch_size;
        if (real_batch_size == 0) {
            printf("No valid turns in episode %d (turns : %d), skipping...\n", ep + 1, turn);
            continue; // No valid turns, skip this episode
        }
        
        for (int batch_start_idx = 0; batch_start_idx < p->size; batch_start_idx += batch_size) {
            int real_batch_size = (p->size - batch_start_idx < batch_size) ? (p->size - batch_start_idx) : batch_size;
            float **inputs = malloc(real_batch_size * sizeof(float*));
            float **targets = malloc(real_batch_size * sizeof(float*));
            for (int i = 0; i < real_batch_size; i++) {
                state *s = p->samples[batch_start_idx + i];
                inputs[i] = s->input;
                network->input = s->input;
                feed_forward(network, ReLU);
                targets[i] = s->reward;
                for (int j = 0; j < 40; j++) {
                    if (isnan(targets[i][j])) {
                        printf("Network output unstable in batch %d, episode %d\n", 1 + batch_start_idx / batch_size, ep + 1);
                        fflush(stdout);
                        exit(1);
                    }
                    targets[i][j] += gamma * network->output[j];
                }
            }
            batched_backpropagate(network, inputs, targets, real_batch_size, ReLU, ReLU_derivative, learning_rate);
        }
        printf("Episode %d finished, total_lr=%d\n", ep+1, total_lines_removed);
        // printf("\n");
        fflush(stdout);
        epsilon -= epsilon_decay;
        free_pool(p);
    }
}

#endif