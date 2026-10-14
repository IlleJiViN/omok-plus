#include "omok_engine.h"

#ifdef _WIN32
#define EXPORT_API extern "C" __declspec(dllexport)
#else
#define EXPORT_API extern "C"
#endif

using namespace omok;

EXPORT_API OmokBoard* create_board() {
    return new OmokBoard();
}

EXPORT_API void free_board(OmokBoard* b) {
    delete b;
}

EXPORT_API void reset_board(OmokBoard* b) {
    b->reset();
}

EXPORT_API int place_stone_c(OmokBoard* b, int r, int c, int8_t player, CaptureRecord* rec) {
    return b->place_stone(r, c, player, *rec);
}

EXPORT_API void undo_stone_c(OmokBoard* b, int r, int c, int8_t player, const CaptureRecord* rec) {
    b->undo_stone(r, c, player, *rec);
}

EXPORT_API int check_game_status_c(const OmokBoard* b) {
    return b->check_game_status();
}

EXPORT_API int get_grid_c(const OmokBoard* b, int8_t* out_grid) {
    std::memcpy(out_grid, b->grid, sizeof(b->grid));
    return 0;
}

EXPORT_API int set_grid_c(OmokBoard* b, const int8_t* in_grid) {
    std::memcpy(b->grid, in_grid, sizeof(b->grid));
    b->move_count = 0;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (b->grid[r][c] != EMPTY) b->move_count++;
        }
    }
    return 0;
}
