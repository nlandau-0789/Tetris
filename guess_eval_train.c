#ifdef GUESS_EVAL_TRAIN_C
bool stop_train = false;
int train(nn *network, __m256i board, float learning_rate, int piece, int n_turns) {
    int next_piece = rand() % 7;
    int max_depth = 0;

    if (n_turns > 1000 || stop_train) {
        stop_train = true;
        return 0; // stop training if too many turns or stop signal
    }

    for (int r = 0; r < n_rot[piece]; r++) {
        __m256i piece_board_init = placements[piece][r];
        for (int x = 0; x < 10-piece_width[piece][r]+1; x++) {
            int y = 16;
            __m256i piece_board = piece_board_init;
            if (!is_zero_m256i(_mm256_and_si256(piece_board, board))){
                piece_board_init = rotate_right_one(piece_board_init);
                continue;
            }
            while (y > piece_height[piece][r] && is_zero_m256i(_mm256_and_si256(_mm256_srli_epi16(piece_board, 1), board))){
                y--;
                piece_board = _mm256_srli_epi16(piece_board, 1);
            }

            __m256i new_board = _mm256_or_si256(board, piece_board);
            remove_full_lines(&new_board);

            int depth = train(network, new_board, learning_rate, next_piece, n_turns + 1);

            calc_consts(new_board);
            short input_short[160];
            for (int i = 0; i < 10; i++) {
                _mm256_storeu_si256((__m256i*)(&input_short[i*16]), consts[i]);
            }
            float input[160];
            for (int i = 0; i < 160; i++) {
                input[i] = (float)input_short[i];
            }

            backpropagate(network, input, 1.0 - exp(-depth/50), ReLU, ReLU_derivative, learning_rate);

            if (depth + 1 > max_depth) {
                max_depth = depth + 1;
            }
            
            piece_board_init = rotate_right_one(piece_board_init);
        }
    }

    if (max_depth > 3) {
        calc_consts(board);
        short input_short[160];
        for (int i = 0; i < 10; i++) {
            _mm256_storeu_si256((__m256i*)(&input_short[i*16]), consts[i]);
        }
        float input[160];
        for (int i = 0; i < 160; i++) {
            input[i] = (float)input_short[i];
        }
        backpropagate(network, input, 1.0 - exp(-max_depth/50), ReLU, ReLU_derivative, learning_rate);
    }
    return max_depth;
}
#endif