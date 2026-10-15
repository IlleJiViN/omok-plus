import sys, time
sys.path.insert(0, 'C:/Users/pc/Downloads')
sys.path.insert(0, 'C:/Users/pc/omok_project')
import torch
import numpy as np
from engine.board import Board
from friend_agent import AgentPlayer, to_rc, to_idx, BoardLightOfBucks
from models.policy_value_net import PolicyValueNet

class GrandmasterAlphaOmok(AgentPlayer):
    """
    Grandmaster AlphaOmok:
    Preserves full mathematical integrity of VCF/VCT deep threat verification,
    while replacing handcrafted heuristic evaluations with the 7.88M ResNet-10 Neural Brain.
    """
    def __init__(self, color):
        super().__init__(color)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load('C:/Users/pc/omok_project/models/checkpoints/rl_model_best.pt', map_location=self.device)
        self.net = PolicyValueNet.create_2x().to(self.device)
        st = ckpt.get('model_state_dict') or ckpt
        self.net.load_state_dict(st)
        self.net.eval()
        self.sim_board = Board()

    def _neural_rank(self, candidates):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        self.sim_board.grid = self.engine.to_external()
        probs, _ = self.net.predict(self.sim_board, self.color, self.device)
        return max(candidates, key=lambda idx: probs[to_rc(idx)[0] * 19 + to_rc(idx)[1]])

    def _tie_search(self, turn, ties):
        # Override tie search with pure ResNet-10 neural policy
        return self._neural_rank(ties)

    def _fallback_pick(self, turn, cands, l1_pass, label, nodes):
        # If there are surviving moves that don't lose immediately, let the neural net pick!
        st = self.stats
        ln = turn.ln
        l1 = turn.l1
        l2 = turn.l2

        if l1_pass:
            # Pick neural best among l1 passing moves
            pick = self._neural_rank(l1_pass)
            st['last_vcf'] = label
            if pick in l2:
                st['last_vct'] = 'unknown' if l2[pick] is None else 'lost'
            return self._done(pick, turn, nodes)

        alive = [m for m in cands if ln.get(m) is False]
        if alive:
            pick = self._neural_rank(alive)
        elif cands:
            pick = self._neural_rank(cands)
        else:
            pick = None

        st['last_vcf'] = 'lost' if label == 'break' else 'checked'
        return self._done(pick, turn, nodes)


def run_match(black_is_alphaomok=True):
    b = Board()
    if black_is_alphaomok:
        p_black = GrandmasterAlphaOmok(0)
        p_white = AgentPlayer(1)
        names = ("Grandmaster AlphaOmok (Black)", "FriendAgent (White)")
    else:
        p_black = AgentPlayer(0)
        p_white = GrandmasterAlphaOmok(1)
        names = ("FriendAgent (Black)", "Grandmaster AlphaOmok (White)")

    print(f"\n{'='*60}")
    print(f"MATCH: {names[0]} vs {names[1]}")
    print(f"{'='*60}")

    curr = 0
    for turn in range(1, 60):
        p = p_black if curr == 0 else p_white
        t_start = time.perf_counter()
        mv = p.take_turn(b.grid, 60000.0)
        dur = (time.perf_counter() - t_start) * 1000.0
        caps = b.place_stone(mv[0], mv[1], curr)
        cap_str = f" [Caps: {len(caps)}]" if caps else ""
        name = "AlphaOmok" if (curr == 0 and black_is_alphaomok) or (curr == 1 and not black_is_alphaomok) else "Friend"
        print(f"Turn {turn:2d}: {name} -> {mv}{cap_str} ({dur:.1f}ms)", flush=True)
        over, winner = b.check_game_status()
        if over:
            w_name = names[winner] if winner in (0, 1) else "Draw"
            print(f"\n>>> GAME OVER at Turn {turn}: WINNER is {w_name}! <<<")
            return winner
        curr = 1 - curr
    return -1


if __name__ == "__main__":
    print("--- Game 1: Grandmaster AlphaOmok (Black) vs FriendAgent (White) ---")
    w1 = run_match(black_is_alphaomok=True)

    print("\n--- Game 2: FriendAgent (Black) vs Grandmaster AlphaOmok (White) ---")
    w2 = run_match(black_is_alphaomok=False)
