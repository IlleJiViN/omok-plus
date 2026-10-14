# Gomoku Dataset Investigation & Feasibility Analysis for Omok++

## 1. External Dataset Audit Summary

| Source | Access | License | Board Size | Format | Labels | Omok++ Usability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Karesis/Gomoku** (HF) | Open (Direct download) | MIT | 15×15 | JSON / PKL / NPY / CSV | Moves only (no explicit outcome) | ❌ **Incompatible** (15x15, standard Gomoku, no capture rule) |
| **PoolC/gomoku-dataset-1.8M** (HF) | Open (Parquet) | Unspecified | 20×20 (tokens 0..399) | Parquet (`input_ids`) | Move sequences only (no win/loss) | ❌ **Incompatible** (20x20, standard Gomoku, no capture rule) |
| **GomokuParty** | Web / Limited | Proprietary | 15×15 | SGF / Web viewer | Moves, outcome | ❌ **Incompatible** (15x15, Renju/Freestyle rules) |
| **Gomocup** | Open (.zip / .psq) | Copyright Gomocup | 20×20 & 15×15 | Piskvork `.psq` logs | Move sequence, engine scores | ❌ **Incompatible** (20x20, standard Gomoku, no captures) |

---

## 2. Incompatibility & Distributional Shift Analysis

1. **Board Dimension Mismatch**:
   - `Karesis` and `GomokuParty` use **15×15** boards (225 positions).
   - `PoolC` and `Gomocup` use **20×20** boards (400 positions).
   - Omok++ strictly requires a standard Go board dimension of **19×19** (361 positions).
   - Embedding 15×15 into 19×19 creates massive edge padding artifacts and completely distorts opening center-of-board play, line-length heuristics, and diagonal geometry.

2. **Crucial Rule Divergence (The Omok++ Capture Mechanic)**:
   - In standard Gomoku / Renju, stones placed on the board are permanent.
   - In **Omok++**, flanking two contiguous opponent stones captures and permanently removes them from the board.
   - Tactical patterns that are safe in standard Gomoku (e.g. adjacent stone pairs without flank protection) are fatal blunders in Omok++.
   - Conversely, creating capture traps and dynamic board vacancies is a fundamental pillar of Omok++ strategy.

3. **Victory Condition Differences**:
   - Omok++ strictly mandates **exact 5-in-a-row**. 6-in-a-row does **NOT** win.
   - In Freestyle Gomoku, 6+ stones is an immediate win.
   - In Renju, Black is restricted from 3-3, 4-4, and overlines, while White can win with overlines.
   - Omok++ allows both players to win if exact 5 is made, and a capture can trigger an opponent win if it trims an overline to 5.

---

## 3. Decision & Synthetic Data Generation Strategy

In accordance with project guidelines:
> "적합한 공개 데이터가 부족하면 기존 엔진을 이용해 Omok++ 전용 합성 데이터를 생성하라. 가능하면 규칙 기반 또는 탐색 기반 에이전트 간 대국으로 초기 데이터를 생성하라."

We generate **native Omok++ self-play / heuristic datasets** using:
1. `HeuristicSearchPlayer` (Agent A: Alpha-Beta Minimax)
2. `SimplePlayer` (tactical heuristic bot with capture awareness)
3. Temperature-based stochastic exploration during the opening (first 4-6 moves) to guarantee diverse board configurations and avoid deterministic loop collapse.
4. Each sample records:
   - `board_state`: Canonical 7-channel 19×19 tensor
   - `current_player`: 0 (Black) or 1 (White)
   - `policy_target`: 361-dimensional probability distribution (or one-hot chosen move)
   - `game_outcome`: +1 (Current player win), -1 (Loss), 0 (Draw)
5. **8-Fold Dihedral Symmetry Data Augmentation (D4 group)**:
   - 4 rotations (0°, 90°, 180°, 270°) × 2 reflections (horizontal flip)
   - 1 game generates 8× unique valid training positions.
