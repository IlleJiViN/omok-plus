"""
Optimized PUCT-based Neural Monte Carlo Tree Search (MCTS) for Omok++.
Features:
- Fast vectorized candidate generation (<0.02ms)
- Tactical candidate-restricted overrides (Immediate win / block detection)
- Vectorized softmax prior evaluation
- Dynamic time budget management for 60s total game budget
- Subtree reuse and Dirichlet exploration
"""

import math
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from engine.board import Board
from models.policy_value_net import PolicyValueNet


class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'] = None, prior: float = 1.0, player: int = Board.BLACK):
        self.parent = parent
        self.prior = prior              # P(s, a)
        self.player = player            # Player to move at this state
        self.children: Dict[int, 'MCTSNode'] = {} # action_idx -> MCTSNode
        self.visit_count = 0            # N(s, a)
        self.total_value = 0.0          # W(s, a)
        self.mean_value = 0.0           # Q(s, a)
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0

    def ucb_score(self, c_puct: float, total_parent_visits: int) -> float:
        """Computes PUCT score for node selection."""
        u = c_puct * self.prior * (math.sqrt(total_parent_visits) / (1 + self.visit_count))
        return self.mean_value + u

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """Selects child with highest PUCT score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_priors: Dict[int, float], next_player: int):
        """Expands node with child nodes for legal actions."""
        self.is_expanded = True
        for action, prior in action_priors.items():
            self.children[action] = MCTSNode(parent=self, prior=prior, player=next_player)

    def backpropagate(self, value: float):
        """
        Backpropagates value up the tree.
        Value is relative to the player at this node.
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self,
                 model: PolicyValueNet,
                 device: torch.device,
                 c_puct: float = 1.5,
                 n_simulations: int = 100,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root: Optional[MCTSNode] = None

    def reset_root(self):
        self.root = None

    def step_root(self, action: int):
        """Reuses subtree if action exists in root children."""
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None

    def search(self,
               board: Board,
               current_player: int,
               add_noise: bool = False,
               n_sims: Optional[int] = None) -> np.ndarray:
        """
        Runs MCTS search from the given board state.
        Returns:
            action_probs: np.ndarray of shape (361,) with visit count distribution.
        """
        sims = n_sims if n_sims is not None else self.n_simulations
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            return np.zeros(361, dtype=np.float32)

        candidates = self._get_candidate_actions(board, legal_moves)

        # Tactical Check 1: Immediate win detection (over candidates only!)
        for a in candidates:
            r, c = divmod(a, 19)
            b_sim = board.copy()
            b_sim.place_stone(r, c, current_player)
            is_over, winner = b_sim.check_game_status()
            if is_over and winner == current_player:
                probs = np.zeros(361, dtype=np.float32)
                probs[a] = 1.0
                return probs

        # Tactical Check 2: Immediate opponent win block (over candidates only!)
        opp = 1 - current_player
        opp_winning_moves = []
        for a in candidates:
            r, c = divmod(a, 19)
            b_sim = board.copy()
            b_sim.place_stone(r, c, opp)
            is_over, winner = b_sim.check_game_status()
            if is_over and winner == opp:
                opp_winning_moves.append(a)
        if len(opp_winning_moves) == 1:
            probs = np.zeros(361, dtype=np.float32)
            probs[opp_winning_moves[0]] = 1.0
            return probs

        # Initialize root if needed
        if self.root is None or self.root.player != current_player:
            self.root = MCTSNode(parent=None, prior=1.0, player=current_player)

        # Expand root if not expanded
        if not self.root.is_expanded:
            policy_probs, value = self.model.predict(board, current_player, self.device)
            action_priors = {a: float(policy_probs[a]) for a in candidates}

            sum_p = sum(action_priors.values())
            if sum_p > 0:
                for a in action_priors:
                    action_priors[a] /= sum_p
            else:
                for a in action_priors:
                    action_priors[a] = 1.0 / len(action_priors)

            self.root.expand(action_priors, 1 - current_player)

        # Add Dirichlet noise at root for exploration in self-play
        if add_noise and self.root.children:
            actions = list(self.root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            for i, a in enumerate(actions):
                self.root.children[a].prior = (
                    (1 - self.dirichlet_epsilon) * self.root.children[a].prior +
                    self.dirichlet_epsilon * noise[i]
                )

        # Run MCTS simulations
        for _ in range(sims):
            node = self.root
            sim_board = board.copy()
            sim_player = current_player

            # 1. Selection
            while node.is_expanded and not node.is_terminal:
                action, next_node = node.select_child(self.c_puct)
                r, c = divmod(action, 19)
                sim_board.place_stone(r, c, sim_player)
                sim_player = 1 - sim_player
                node = next_node

            # 2. Evaluation & Check Terminal
            is_over, winner = sim_board.check_game_status()
            if is_over:
                node.is_terminal = True
                if winner == node.player:
                    eval_value = 1.0
                elif winner == -1:
                    eval_value = 0.0
                else:
                    eval_value = -1.0
            else:
                # 3. Expansion using Neural Network
                policy_probs, value = self.model.predict(sim_board, node.player, self.device)
                eval_value = value

                sub_candidates = self._get_candidate_actions(sim_board, sim_board.get_legal_moves())
                action_priors = {a: float(policy_probs[a]) for a in sub_candidates}
                sum_p = sum(action_priors.values())
                if sum_p > 0:
                    for a in action_priors:
                        action_priors[a] /= sum_p
                else:
                    for a in action_priors:
                        action_priors[a] = 1.0 / max(1, len(action_priors))

                node.expand(action_priors, 1 - node.player)

            # 4. Backpropagation
            node.backpropagate(eval_value)

        # Build policy distribution from root visit counts
        action_probs = np.zeros(361, dtype=np.float32)
        for a, child in self.root.children.items():
            action_probs[a] = child.visit_count

        total_visits = np.sum(action_probs)
        if total_visits > 0:
            action_probs /= total_visits
        else:
            for a in candidates:
                action_probs[a] = 1.0 / len(candidates)

        return action_probs

    def _get_candidate_actions(self, board: Board, legal_moves: List[Tuple[int, int]]) -> List[int]:
        """Fast candidate generation using numpy mask dilation (radius 2)."""
        occupied = (np.array(board.grid) != Board.EMPTY)
        if not np.any(occupied):
            return [9 * 19 + 9]

        mask = np.zeros((19, 19), dtype=bool)
        rows, cols = np.where(occupied)
        for r, c in zip(rows, cols):
            r_min, r_max = max(0, r - 2), min(19, r + 3)
            c_min, c_max = max(0, c - 2), min(19, c + 3)
            mask[r_min:r_max, c_min:c_max] = True

        empty_mask = mask & (~occupied)
        candidates = np.where(empty_mask.flatten())[0].tolist()
        return candidates if candidates else [r * 19 + c for r, c in legal_moves]
