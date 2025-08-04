#ifdef Q_LEARNING_TRAIN_C
#ifndef NN_INPUT_SIZE
#define NN_INPUT_SIZE 100
#endif
#define MAX_TURNS 2000000000
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
    for (int s = 0; s < seasons; s++) {
        float epsilon = epsilon_i;
        float gamma = gamma_i;
        int rew_idx = s % n_phases;
        int total_lines_removed = 0;
        double avg_output = 0.0;
        // float threshold_lines = 1000;
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
                if (isnan(network->output) || fabs(network->output) > 1e15) {
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
            }
            if (turn == 0) {
                get_placement(last_piece, board, network);
            }
            if ((ep + 1) % 500 == 0) {
                printf("\rEpisode %d finished, avg_lr=%lf, avg_output=%lf        ", ep+1, (double)total_lines_removed/500, avg_output / 500.0);
                avg_output = 0.0;
                fflush(stdout);
                total_lines_removed = 0;
            } 
            epsilon *= 0.9999;
            // gamma *= 1.0004;
            // if (epsilon < 0.15) {
            //     epsilon = 0.15f; // minimum epsilon
            // }
            // if (gamma > 0.99) {
            //     gamma = 0.99; // maximum gamma
            // }
        }
        // if (total_lines_removed > threshold_lines * episodes) {
        //     learning_rate *= 0.99;
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
        printf("\n");
    }
}

void batched_q_train(nn* network, int n_episodes, int batch_size, float learning_rate, float gamma, float epsilon, float epsilon_decay, reward_t rew) {
    float **states_memory = malloc(MAX_TURNS * sizeof(float*));
    float *targets_memory = malloc(MAX_TURNS * sizeof(float));
    float **states_batch = malloc(batch_size * sizeof(float*));
    float *targets_batch = malloc(batch_size * sizeof(float));
    for (int i = 0; i < MAX_TURNS; i++) {
        states_memory[i] = malloc(NN_INPUT_SIZE * sizeof(float));
    }
    
    bool *seen_idx = malloc(MAX_TURNS * sizeof(int));
    int memory_size;
    for (int ep = 0; ep < n_episodes; ep++){
        memory_size = 0;
        __m256i board = ZERO;
        int piece = rand() % 7;
        int turn = 0;
        int n_lines_removed = 0, total_lines_removed = 0;
        while (turn < MAX_TURNS) {
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
                reward = -1.0f;
                float target = -1.0f;

                float input[NN_INPUT_SIZE];
                get_nn_input(input, board);
                backpropagate(network, input, target, ReLU, ReLU_derivative, learning_rate);
                break;
            }
            
            __m256i new_board = place(piece, best.x, best.r, board, &n_lines_removed);
            reward = rew(board, new_board, n_lines_removed);
            total_lines_removed += n_lines_removed;

            // Prepare NN input for current state
            float next_input[NN_INPUT_SIZE];
            get_nn_input(states_memory[memory_size], board);
            get_nn_input(next_input, new_board);

            network->input = next_input;
            feed_forward(network, ReLU);
            if (isnan(network->output) || fabs(network->output) > 1e6) {
                printf("Network output unstable: %f\n", network->output);
                fflush(stdout);
                exit(1);
            }
            float next_value = (network->output);

            float target = (reward + gamma * next_value);
            // if (target > 1000.0f) target = 1000.0f;
            // if (target < -1000.0f) target = -1000.0f;

            targets_memory[memory_size] = target;
            memory_size++;
            turn++;
            // printf("\rTurn %d, piece %d, x=%d, r=%d, reward=%.2f, total_lines_removed=%d", turn, piece, best.x, best.r, reward, total_lines_removed);
            board = new_board; 
        }
        // printf("\n");
        
        int real_batch_size = memory_size < batch_size ? memory_size : batch_size;
        if (real_batch_size == 0) {
            printf("No valid turns in episode %d (turns : %d), skipping...\n", ep + 1, turn);
            continue; // No valid turns, skip this episode
        }

        // printf("Game done, starting backpropagation for episode %d\n", ep + 1);
        
        for (int epoch = 0; epoch < 3; epoch++) {
            for (int i = 0; i < memory_size; i++) {
                seen_idx[i] = false; // Initialize seen indices
            }
            for (int i = 0; i < real_batch_size; i++) {
                int idx = rand() % memory_size;
                while (seen_idx[idx]) {
                    idx = rand() % memory_size;
                }
                seen_idx[idx] = true;
                states_batch[i] = states_memory[idx];
                targets_batch[i] = targets_memory[idx];
            }
            
            // printf("Epoch %d\n", epoch + 1);
            batched_backpropagate(network, states_batch, targets_batch, real_batch_size, ReLU, ReLU_derivative, learning_rate);
        }
        printf("\rEpisode %d finished, total_lr=%d        ", ep+1, total_lines_removed);
        // printf("\n");
        fflush(stdout);
        epsilon -= epsilon_decay;
    }
    
    for (int i = 0; i < MAX_TURNS; i++) {
        free(states_memory[i]);
    }
    free(targets_memory);
    free(states_memory);
    free(states_batch);
    free(targets_batch);
    free(seen_idx);
}

#endif