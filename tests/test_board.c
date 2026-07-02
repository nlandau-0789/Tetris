#include "v5.c"

int main() {
    // bool *t = aligned_alloc(32, 160 * sizeof(bool));
    bool *t = malloc(160 * sizeof(bool));
    if (!t) {
        perror("aligned_alloc failed");
        return 1;
    }
    for (int i = 0; i < 160; i++){
        t[i] = false;
    }
    for (int i = 0; i < 8; i++){
        t[i] = true;
    }
    for (int i = 2; i < 6; i++){
        t[10*i+3] = true;
    }
    for (int i = 4; i < 6; i++){
        t[10*i+4] = true;
    }
    for (int i = 0; i < 14; i+=2){
        t[10*i+8] = true;
    }
    for (int i = 0; i < 4; i++){
        t[10*i] = true;
    }
    for (int i = 0; i < 4; i++){
        t[10*i+2] = true;
    }
    t[159] = true;
    __m256i board = board_from_array(t);
    print_board(board);

    print_features(board);

    for (int i = 0; i < 10; i++) {
        t[10*3 + i] = true;
        t[10*4 + i] = true;
        t[10*7 + i] = true;
    }
    board = board_from_array(t);
    remove_full_lines(&board);
    print_board(board);

    return 0;
}