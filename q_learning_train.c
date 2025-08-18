#ifndef NN_INPUT_SIZE
#define NN_INPUT_SIZE 100
#endif

#ifndef NN_OUTPUT_SIZE
#define NN_OUTPUT_SIZE 40
#endif

#ifndef MAX_TURNS
#define MAX_TURNS 2000000000
#endif

#ifdef REWARD_FUNCS
float max_reward = -INFINITY;
float min_reward = INFINITY;
float max_target = -INFINITY;
float min_target = INFINITY;
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
    // Temporary arrays to safely read __m256i
    uint16_t heights_new[16], holes_new_arr[16];
    uint16_t heights_old[16], holes_old_arr[16];

    // Compute constants for new board
    calc_consts(new_board);
    _mm256_storeu_si256((__m256i*)heights_new, consts[0]);
    _mm256_storeu_si256((__m256i*)holes_new_arr, consts[2]);

    short holes_new = 0;
    for (int i = 0; i < 10; i++) {
        holes_new += holes_new_arr[i];
    }
    short max_height_new = ((short*)&consts[1])[0];

    // Compute constants for old board
    calc_consts(old_board);
    _mm256_storeu_si256((__m256i*)heights_old, consts[0]);
    _mm256_storeu_si256((__m256i*)holes_old_arr, consts[2]);

    short holes_old = 0;
    for (int i = 0; i < 10; i++) {
        holes_old += holes_old_arr[i];
    }
    short max_height_old = ((short*)&consts[1])[0];

    float new_holes = holes_new - holes_old;
    float max_height_diff = max_height_new - max_height_old;

    // Example reward formula
    float reward = (float)n_lines_removed * 100.0f - new_holes * 0.4f - max_height_diff * 0.2f;
    reward /= 50.0f;
    reward += 0.5f;

    if (reward > max_reward) max_reward = reward;
    if (reward < min_reward) min_reward = reward;

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
#endif

#ifdef Q_LEARNING_TRAIN_C
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
    float *s;
    float Q;
    int a;
};

typedef struct state state;

void free_state(void *st){
    state *s = (state*)st;
    free(s->s);
    free(s);
}

void batched_q_train(nn* online, nn* target, int n_episodes, int n_batches, int batch_size, float learning_rate, float gamma, float epsilon, float epsilon_decay, reward_t rew) {
    pool *p;
    bool valid_placements[NN_OUTPUT_SIZE];

    for (int ep = 0; ep < n_episodes; ep++) {
        p = init_pool(n_batches * batch_size, free_state, rand());
        __m256i board = ZERO;
        int turn = 0;
        int n_lines_removed = 0, total_lines_removed = 0;
        float max_target = -INFINITY;

        while (turn < MAX_TURNS) {
            int piece = rand() % 7;
            int n_valid_placements = get_valid_placements(valid_placements, piece, board);

            state *s = malloc(sizeof(state));
            s->s = malloc(sizeof(float) * NN_INPUT_SIZE);
            s->a = -1;
            s->Q = 0.0f;
            
            get_nn_input(s->s, board);

            if (n_valid_placements == 0) {
                add_to_pool(p, s);
                break;
            }

            int r = 0, x = 0;

            // --- Action selection: use ONLINE network ---
            online->input = s->s;
            feed_forward(online, ReLU);
            for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
                if (isnan(online->output[i])) {
                    printf("Network output unstable\n");
                    print_nn(online);
                    printf("Input: ");
                    for (int j = 0; j < NN_INPUT_SIZE; j++) {
                        printf("%f ", online->input[j]);
                    }
                    printf("\n");
                    print_board(board);
                    exit(1);
                }
            }

            // --- Epsilon-greedy ---
            if (((float)rand() / RAND_MAX) < epsilon) {
                // Random action
                int idx = rand() % n_valid_placements;
                int valid_move_count = 0;
                for (int rr = 0; rr < n_rot[piece]; rr++) {
                    for (int xx = 0; xx < (10 - piece_width[piece][rr] + 1); xx++) {
                        if (valid_placements[rr * 10 + xx]) {
                            if (valid_move_count == idx) {
                                r = rr; 
                                x = xx;
                                s->a = rr * 10 + xx;
                            }
                            valid_move_count++;
                        }
                    }
                }
            } else {
                // Greedy action from ONLINE network
                float best_eval = -INFINITY;
                for (int rr = 0; rr < n_rot[piece]; rr++) {
                    for (int xx = 0; xx < (10 - piece_width[piece][rr] + 1); xx++) {
                        if (valid_placements[rr * 10 + xx]) {
                            if (online->output[rr * 10 + xx] > best_eval) {
                                r = rr; 
                                x = xx;
                                s->a = rr * 10 + xx;
                                best_eval = online->output[rr * 10 + xx];
                            }
                        }
                    }
                }
                max_target = fmaxf(max_target, best_eval);
            }

            if (s->a == -1) {
                printf("No valid action found, piece %d, board: ", piece);
                print_board(board);
                exit(1);
            }

            // --- Step environment ---
            __m256i new_board = place(piece, x, r, board, &n_lines_removed);
            float reward = rew(board, new_board, n_lines_removed);
            total_lines_removed += n_lines_removed;

            // --- Bootstrap with TARGET network ---
            get_nn_input(target->input, new_board);
            feed_forward(target, ReLU);

            float max_next_q = 0.0f;
            for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
                if (!isnan(target->output[j]) && target->output[j] > max_next_q) {
                    max_next_q = target->output[j];
                }
            }

            // TD target for the chosen action
            s->Q = reward + gamma * max_next_q;
            // printf("reward: %f, max_next_q: %f\n", reward, max_next_q);

            // Store transition
            add_to_pool(p, s);

            turn++;
            board = new_board;
        }

        // --- Batch training on stored samples ---
        for (int batch_start_idx = 0; batch_start_idx < p->size; batch_start_idx += batch_size) {
            int real_batch_size = (p->size - batch_start_idx < batch_size)
                                  ? (p->size - batch_start_idx)
                                  : batch_size;
            float **inputs  = malloc(real_batch_size * sizeof(float*));
            float **targets = malloc(real_batch_size * sizeof(float*));
            for (int i = 0; i < real_batch_size; i++) {
                targets[i] = malloc(sizeof(float) * NN_OUTPUT_SIZE);
                state *s = p->samples[batch_start_idx + i];
                inputs[i] = s->s ;
                online->input = inputs[i];
                feed_forward(online, ReLU);
                if (s->a == -1) {
                    for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
                        targets[i][j] = 0.0f;
                    }
                } else {
                    for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
                        targets[i][j] = online->output[j];
                    }
                    targets[i][s->a] = s->Q; // Update target for the chosen action
                }
            }
            batched_backpropagate(online, inputs, targets, real_batch_size, ReLU, ReLU_derivative, learning_rate);
            for (int i = 0; i < real_batch_size; i++) {
                free(targets[i]);
            }
            free(inputs);
            free(targets);
        }

        // --- Soft update: ONLINE → TARGET ---
        float tau = 1.0f;
        weight_avg_nn(target, online, 1.0f - tau);

        printf("Episode %d finished, total_lr=%d, rew_range = [%f:%f], max_t = %f\n", ep+1, total_lines_removed, min_reward, max_reward, max_target);
        fflush(stdout);

        epsilon -= epsilon_decay;
        free_pool(p);
    }
}


#endif