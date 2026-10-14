"""
Generates class-diagram.pdf conforming to CAS 1102 Project 1 Section 1.1:
- Two-slotted boxes with class names
- Upper slot: member variables (type, visibility, description)
- Lower slot: methods (inputs, outputs, description)
- Abstract classes drawn with dotted boundary
- Solid line with arrow for inheritance
- Dotted line for instantiation/aggregation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages


def create_class_diagram(output_pdf: str = "submission/class-diagram.pdf"):
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 120)
    ax.axis('off')

    def draw_class_box(x, y, w, h, name, vars_text, methods_text, is_abstract=False):
        # Header + Box
        linestyle = '--' if is_abstract else '-'
        edgecolor = '#2c3e50'
        facecolor = '#f8f9fa' if not is_abstract else '#edf2f7'

        # Main box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                      edgecolor=edgecolor, facecolor=facecolor,
                                      linestyle=linestyle, linewidth=1.5)
        ax.add_patch(rect)

        # Header bar
        header_h = 4.5
        ax.text(x + w / 2, y + h - 2.8, name, fontsize=10, fontweight='bold',
                ha='center', va='center', color='#1a365d')

        # Divider 1
        ax.plot([x, x + w], [y + h - header_h, y + h - header_h], color=edgecolor, linewidth=1)

        # Variables slot
        split_y = y + (h - header_h) * 0.52
        ax.text(x + 1.5, y + h - header_h - 1.5, vars_text, fontsize=7.5,
                ha='left', va='top', family='monospace', color='#2d3748')

        # Divider 2
        ax.plot([x, x + w], [split_y, split_y], color=edgecolor, linewidth=1)

        # Methods slot
        ax.text(x + 1.5, split_y - 1.5, methods_text, fontsize=7.5,
                ha='left', va='top', family='monospace', color='#2d3748')

    # 1. Player (Abstract)
    draw_class_box(
        x=10, y=75, w=38, h=22,
        name="«Abstract» Player",
        vars_text="+ player_id: int\n  (0: Black, 1: White)",
        methods_text="+ take_turn(board, time) -> (r,c)\n  [abstract] compute next move",
        is_abstract=True
    )

    # 2. Board
    draw_class_box(
        x=60, y=75, w=44, h=38,
        name="Board",
        vars_text="+ grid: List[List[int]] (19x19 state)\n+ captures: Dict[int, int]\n+ last_move: Optional[Tuple[int,int]]\n+ move_history: List[Tuple]",
        methods_text="+ place_stone(r, c, p) -> List[Tuple]\n+ count_exact_fives(player) -> int\n+ check_game_status() -> (bool, winner)\n+ get_legal_moves() -> List[Tuple]\n+ is_full() -> bool\n+ display() -> str"
    )

    # 3. GameManager
    draw_class_box(
        x=115, y=75, w=40, h=30,
        name="GameManager",
        vars_text="+ players: Dict[int, Player]\n+ board: Board\n+ remaining_time: Dict[int, float]\n+ time_limit_ms: float",
        methods_text="+ play_game() -> (winner, reason)\n  (alternates turns, checks timeouts,\n   applies captures, declares victory)"
    )

    # 4. Concrete Players
    # RandomPlayer
    draw_class_box(
        x=5, y=40, w=30, h=16,
        name="RandomPlayer",
        vars_text="",
        methods_text="+ take_turn(board, time)\n  returns uniform random empty cell"
    )

    # SimplePlayer
    draw_class_box(
        x=40, y=40, w=35, h=20,
        name="SimplePlayer",
        vars_text="+ opponent_id: int",
        methods_text="+ take_turn(board, time)\n+ _evaluate_move(b, r, c)\n+ _score_line(b, r, c, dr, dc, p)\n  (greedy win, block, capture)"
    )

    # HeuristicSearchPlayer
    draw_class_box(
        x=80, y=40, w=38, h=24,
        name="HeuristicSearchPlayer",
        vars_text="+ max_depth: int = 2\n+ time_budget_ms: float",
        methods_text="+ take_turn(board, time)\n+ _min_val(b, d, alpha, beta)\n+ _max_val(b, d, alpha, beta)\n+ _evaluate_board(b)\n  (Alpha-Beta minimax search)"
    )

    # SmartPlayer
    draw_class_box(
        x=123, y=36, w=34, h=28,
        name="SmartPlayer",
        vars_text="+ model: PolicyValueNet\n+ mcts: MCTS\n+ device: torch.device",
        methods_text="+ take_turn(board, time)\n+ _decide_move(b, time)\n+ _fast_policy_move(b, legal)\n+ _emergency_fallback(b)\n  (Neural MCTS + tactical filter)"
    )

    # 5. AI Engine Core: PolicyValueNet & MCTS
    draw_class_box(
        x=45, y=5, w=45, h=26,
        name="PolicyValueNet (nn.Module)",
        vars_text="+ conv_in: Conv2d + BatchNorm\n+ res_blocks: List[ResidualBlock]\n+ p_fc: Linear -> 361 logits\n+ v_fc2: Linear -> 1 value [-1, 1]",
        methods_text="+ forward(x) -> (logits, value)\n+ encode_board(b, p) -> np.ndarray\n+ predict(b, p, dev) -> (probs, val)"
    )

    draw_class_box(
        x=100, y=5, w=50, h=26,
        name="MCTS & MCTSNode",
        vars_text="+ root: Optional[MCTSNode]\n+ c_puct: float, n_simulations: int\n+ Node: P, N, W, Q, children",
        methods_text="+ search(b, p, noise, sims) -> pi\n+ step_root(action)\n+ select_child(c_puct)\n+ expand(priors, next_p)\n+ backpropagate(value)"
    )

    # Arrows for Inheritance (Solid line with arrow)
    arrow_props = dict(arrowstyle='->', lw=1.5, color='#2c3e50')
    # Player -> Concrete players
    ax.annotate('', xy=(20, 56), xytext=(20, 75), arrowprops=arrow_props)
    ax.annotate('', xy=(55, 60), xytext=(35, 75), arrowprops=arrow_props)
    ax.annotate('', xy=(95, 64), xytext=(48, 80), arrowprops=arrow_props)
    ax.annotate('', xy=(135, 64), xytext=(48, 85), arrowprops=arrow_props)

    # Instantiation / Aggregation (Dotted line with arrow)
    dotted_props = dict(arrowstyle='->', lw=1.2, ls='--', color='#e74c3c')
    # GameManager has Board and Players
    ax.annotate('', xy=(104, 90), xytext=(115, 90), arrowprops=dotted_props)
    ax.annotate('', xy=(48, 90), xytext=(115, 100), arrowprops=dotted_props)

    # SmartPlayer uses MCTS and PolicyValueNet
    ax.annotate('', xy=(125, 31), xytext=(135, 36), arrowprops=dotted_props)
    ax.annotate('', xy=(75, 31), xytext=(125, 45), arrowprops=dotted_props)

    # Title
    ax.text(80, 116, "CAS 1102 Project 1: Omok++ Class Architecture Diagram",
            fontsize=15, fontweight='bold', ha='center', color='#1a365d')
    ax.text(80, 113, "Solid arrows: Inheritance (extends Player) | Red dashed arrows: Instantiation / Association",
            fontsize=9, style='italic', ha='center', color='#718096')

    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated class diagram PDF: {output_pdf}")


if __name__ == '__main__':
    create_class_diagram()
