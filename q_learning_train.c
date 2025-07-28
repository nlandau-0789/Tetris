#ifdef Q_LEARNING_TRAIN_C
float max_reward = -INFINITY;
float min_reward = INFINITY;
float phase2_rew(__m256i old_board, __m256i new_board, int n_lines_removed) {
    float reward = (float)n_lines_removed / 4.0f;
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
    float line_reward = 0.0f;
    if (n_lines_removed == 1) line_reward = 50.0f;
    else if (n_lines_removed == 2) line_reward = 90.0f;
    else if (n_lines_removed == 3) line_reward = 100.0f;
    else if (n_lines_removed == 4) line_reward = 110.0f;

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

typedef float (*reward_t)(__m256i, __m256i, int);

void rl_train(nn *network, int episodes, int seasons, float learning_rate, float gamma_i, float epsilon_i, reward_t reward_calcs[], int n_phases) {
    for (int s = 0; s < seasons; s++) {
        float epsilon = epsilon_i;
        float gamma = gamma_i;
        int rew_idx = s % n_phases;
        int total_lines_removed = 0;
        float threshold_lines = 1000;
        for (int ep = 0; ep < episodes; ep++) {
            // Start with an empty board
            __m256i board = ZERO;
            float total_reward = 0;
            int turn = 0;
            int last_piece;
            
            while (turn < 2500) {
                int piece = rand() % 7;
                last_piece = piece;
    
                // Epsilon-greedy: random move or best move
                placement_and_status best;
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
                    reward = -100;
                    total_reward += reward;
                    float target = -100;
    
                    float input[160];
                    get_nn_input(input, board);
                    backpropagate(network, input, target, ReLU, ReLU_derivative, learning_rate);
                    break;
                }
                int n_lines_removed;
                __m256i new_board = place(piece, best.x, best.r, board, &n_lines_removed);
                reward = reward_calcs[rew_idx](board, new_board, n_lines_removed);
                total_reward += reward;
                total_lines_removed += n_lines_removed;
    
                // Prepare NN input for current state
                float input[160], next_input[160];
                get_nn_input(input, board);
                get_nn_input(next_input, new_board);
    
                // Q-learning target: reward + gamma * max_a' Q(next_state, a')
                network->input = next_input;
                if (isnan(network->output) || fabs(network->output) > 1e6) {
                    printf("Network output unstable: %f\n", network->output);
                    exit(1);
                }
                feed_forward(network, ReLU);
                float next_value = (network->output);
    
                float target = (reward + gamma * next_value);
                if (target > 1000.0f) target = 1000.0f;
                if (target < -1000.0f) target = -1000.0f;
    
                // Train on (state, target)
                backpropagate(network, input, target, ReLU, ReLU_derivative, learning_rate);
    
                // Move to next state
                board = new_board;
                turn++;
            }
            if (turn == 0) {
                logged_get_placement(last_piece, board, network);
            }
            if ((ep + 1) % 500 == 0) {
                printf("\rEpisode %d finished, min=%f, max=%f, total_lr=%d        ", ep+1, min_reward, max_reward, total_lines_removed);
                fflush(stdout);
            } 
            epsilon *= 0.9999;
            gamma *= 1.0004;
            if (epsilon < 0.1) {
                epsilon = 0.1; // minimum epsilon
            }
            if (gamma > 0.99) {
                gamma = 0.99; // maximum gamma
            }
        }
        if (total_lines_removed > threshold_lines * episodes) {
            learning_rate *= 0.9;
            threshold_lines *= 3;
            epsilon_i *= 0.95;
            gamma_i *= 1.05;
            if (epsilon_i < 0.1) {
                epsilon_i = 0.1; // minimum epsilon
            }
            if (gamma_i > 0.99) {
                gamma_i = 0.99; // maximum gamma
            }
        }
        printf("\n");

    }
}
#endif