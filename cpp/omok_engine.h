#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <tuple>

namespace omok {

constexpr int BOARD_SIZE = 19;
constexpr int TOTAL_CELLS = 361;

constexpr int8_t EMPTY = -1;
constexpr int8_t BLACK = 0;
constexpr int8_t WHITE = 1;

struct Coord {
    int8_t r, c;
};

struct CaptureRecord {
    Coord captured[8]; // Max 8 pairs = 16 stones, stored as pairs
    int8_t count;
};

// 8 directions for custodial capture: (dr, dc)
constexpr int8_t DIRECTIONS[8][2] = {
    {0, 1}, {0, -1}, {1, 0}, {-1, 0},
    {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
};

// 4 main axes for 5-in-a-row: Horizontal, Vertical, Diagonal, Anti-Diagonal
constexpr int8_t AXES[4][2] = {
    {0, 1}, {1, 0}, {1, 1}, {1, -1}
};

class OmokBoard {
public:
    int8_t grid[BOARD_SIZE][BOARD_SIZE];
    int captures[2]; // captures[0]: Black captures, captures[1]: White captures
    int move_count;

    OmokBoard() {
        reset();
    }

    void reset() {
        std::memset(grid, EMPTY, sizeof(grid));
        captures[0] = 0;
        captures[1] = 0;
        move_count = 0;
    }

    inline bool is_valid_coord(int r, int c) const {
        return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
    }

    inline bool is_empty(int r, int c) const {
        return grid[r][c] == EMPTY;
    }

    // High-speed In-Place Stone Placement with Custodial Captures
    int place_stone(int r, int c, int8_t player, CaptureRecord& record) {
        grid[r][c] = player;
        move_count++;
        int8_t opp = 1 - player;
        record.count = 0;

        for (int i = 0; i < 8; ++i) {
            int8_t dr = DIRECTIONS[i][0];
            int8_t dc = DIRECTIONS[i][1];

            int r1 = r + dr, c1 = c + dc;
            int r2 = r + 2 * dr, c2 = c + 2 * dc;
            int r3 = r + 3 * dr, c3 = c + 3 * dc;

            if (is_valid_coord(r3, c3)) {
                if (grid[r1][c1] == opp && grid[r2][c2] == opp && grid[r3][c3] == player) {
                    // Capture two opponent stones
                    grid[r1][c1] = EMPTY;
                    grid[r2][c2] = EMPTY;
                    record.captured[record.count++] = {static_cast<int8_t>(r1), static_cast<int8_t>(c1)};
                    record.captured[record.count++] = {static_cast<int8_t>(r2), static_cast<int8_t>(c2)};
                    captures[player] += 2;
                }
            }
        }
        return record.count;
    }

    // High-speed In-Place Move Rollback (Zero Memory Allocation)
    void undo_stone(int r, int c, int8_t player, const CaptureRecord& record) {
        int8_t opp = 1 - player;
        grid[r][c] = EMPTY;
        move_count--;
        for (int i = 0; i < record.count; ++i) {
            grid[record.captured[i].r][record.captured[i].c] = opp;
        }
        captures[player] -= record.count;
    }

    // Exact 5-in-a-row victory & simultaneous tie check
    // Returns: -2: Ongoing, -1: Tie, 0: Black Wins, 1: White Wins
    int check_game_status() const {
        bool black_has_5 = false;
        bool white_has_5 = false;

        for (int axis = 0; axis < 4; ++axis) {
            int8_t dr = AXES[axis][0];
            int8_t dc = AXES[axis][1];

            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    int8_t val = grid[r][c];
                    if (val == EMPTY) continue;

                    // Ensure this is the start of a consecutive run
                    int pr = r - dr, pc = c - dc;
                    if (is_valid_coord(pr, pc) && grid[pr][pc] == val) continue;

                    int cnt = 0;
                    int cr = r, cc = c;
                    while (is_valid_coord(cr, cc) && grid[cr][cc] == val) {
                        cnt++;
                        cr += dr;
                        cc += dc;
                    }

                    // Exact 5-in-a-row check
                    if (cnt == 5) {
                        if (val == BLACK) black_has_5 = true;
                        else white_has_5 = true;
                    }
                }
            }
        }

        if (black_has_5 && white_has_5) return -1; // Tie
        if (black_has_5) return 0; // Black
        if (white_has_5) return 1; // White
        if (move_count >= TOTAL_CELLS) return -1; // Full board tie
        return -2; // Ongoing
    }
};

} // namespace omok
