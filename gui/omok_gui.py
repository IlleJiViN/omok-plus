"""
Omok++ Interactive GUI Arena
Play against AlphaZero 2x Neural MCTS AI, Minimax Heuristic Bot, or Simple Tactical Bot.

Features:
- Official 19x19 Omok++ board with wooden aesthetic, star points, and coordinates.
- Realistic 3D stone rendering with specular reflection and last-move indicator.
- Real-time Custodial Capture animation and captured stones scoreboard.
- Strict 60.0s tournament clock countdown per player.
- Multi-threaded non-blocking AI computation with status indicators.
- Multiple AI opponents: SmartPlayer (2x AlphaZero), RLMCTSPlayer, Minimax, SimplePlayer.
- Undo, Hint, and Rules guide modal.
"""

import os
import sys
import time
import math
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Tuple, List, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.board import Board


class OmokGUI:
    # Color palette
    BG_DARK = "#1E1E2E"
    PANEL_BG = "#2A2A3C"
    TEXT_LIGHT = "#E0E0E8"
    TEXT_MUTED = "#8E8EA0"
    BOARD_WOOD = "#DCB35C"
    BOARD_WOOD_LIGHT = "#E5BF6E"
    GRID_LINE = "#4A3B22"
    STAR_POINT = "#3A2A12"
    BLACK_STONE = "#161616"
    BLACK_SHADOW = "#080808"
    WHITE_STONE = "#F5F5FA"
    WHITE_SHADOW = "#D0D0DC"
    HIGHLIGHT_RED = "#FF3366"
    HIGHLIGHT_GOLD = "#FFD700"
    BTN_ACCENT = "#5E5CE6"
    BTN_SUCCESS = "#34C759"
    BTN_WARNING = "#FF9F0A"

    STAR_POINTS = [
        (3, 3), (3, 9), (3, 15),
        (9, 3), (9, 9), (9, 15),
        (15, 3), (15, 9), (15, 15)
    ]
    COL_NAMES = [chr(ord('A') + i) for i in range(19)]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Omok++ AI Arena — AlphaZero 2x Edition")
        self.root.configure(bg=self.BG_DARK)
        self.root.minsize(1040, 760)

        # Game state
        self.board = Board()
        self.current_player = Board.BLACK
        self.human_player = Board.BLACK  # Black moves first
        self.ai_player_id = Board.WHITE
        self.ai_mode = "SmartPlayer (AlphaZero 2x Ensemble)"
        self.ai_instance = None
        self.ai_thinking = False
        self.game_over = False
        self.winner = None

        # Tournament Clocks (60.0s total per player)
        self.TOTAL_TIME = 60.0
        self.time_left = {Board.BLACK: self.TOTAL_TIME, Board.WHITE: self.TOTAL_TIME}
        self.timer_active = False
        self.last_timer_tick = 0.0

        # UI parameters
        self.cell_size = 34
        self.margin = 44
        self.board_pixel_size = self.margin * 2 + self.cell_size * 18

        # Undo stack: list of Board copies
        self.history: List[Board] = []

        # Build UI layout
        self._setup_styles()
        self._create_widgets()
        self._init_ai_player()

        # Canvas event bindings
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        # Start timer loop
        self.root.after(100, self._timer_tick)
        self._redraw_board()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=self.BG_DARK)
        style.configure("Panel.TFrame", background=self.PANEL_BG)
        style.configure("Header.TLabel", background=self.PANEL_BG, foreground=self.TEXT_LIGHT, font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel", background=self.PANEL_BG, foreground=self.TEXT_MUTED, font=("Segoe UI", 9))
        style.configure("Score.TLabel", background=self.PANEL_BG, foreground=self.TEXT_LIGHT, font=("Segoe UI", 11, "bold"))
        style.configure("TCombobox", fieldbackground=self.PANEL_BG, background="#3B3B52", foreground="#FFFFFF")

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, style="Dark.TFrame", padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Board Canvas
        board_container = ttk.Frame(main_frame, style="Dark.TFrame")
        board_container.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas = tk.Canvas(
            board_container,
            width=self.board_pixel_size,
            height=self.board_pixel_size,
            bg=self.BOARD_WOOD,
            highlightthickness=2,
            highlightbackground="#221C12",
            cursor="crosshair"
        )
        self.canvas.pack()

        # Right Column: Side Control Panel
        side_panel = tk.Frame(main_frame, bg=self.PANEL_BG, padx=16, pady=16, width=320)
        side_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        side_panel.pack_propagate(False)

        # Title Card
        title_lbl = tk.Label(
            side_panel,
            text="OMOK++ AI ARENA",
            font=("Segoe UI", 16, "bold"),
            bg=self.PANEL_BG,
            fg="#FFFFFF"
        )
        title_lbl.pack(anchor="w", pady=(0, 2))

        subtitle_lbl = tk.Label(
            side_panel,
            text="AlphaZero 2x Neural MCTS (10 ResBlocks / 128 Channels)",
            font=("Segoe UI", 8),
            bg=self.PANEL_BG,
            fg=self.HIGHLIGHT_GOLD
        )
        subtitle_lbl.pack(anchor="w", pady=(0, 14))

        # Dashboard / Scoreboard Frame
        score_frame = tk.Frame(side_panel, bg="#1E1E2A", bd=1, relief=tk.SOLID, padx=12, pady=10)
        score_frame.pack(fill=tk.X, pady=(0, 14))

        # Black status row
        self.black_row = tk.Frame(score_frame, bg="#1E1E2A")
        self.black_row.pack(fill=tk.X, pady=4)
        tk.Label(self.black_row, text="● 흑 (Black)", font=("Segoe UI", 11, "bold"), bg="#1E1E2A", fg="#FFFFFF").pack(side=tk.LEFT)
        self.black_time_lbl = tk.Label(self.black_row, text="60.0s", font=("Consolas", 12, "bold"), bg="#1E1E2A", fg="#34C759")
        self.black_time_lbl.pack(side=tk.RIGHT)
        self.black_cap_lbl = tk.Label(self.black_row, text="따냄: 0", font=("Segoe UI", 9), bg="#1E1E2A", fg=self.TEXT_MUTED)
        self.black_cap_lbl.pack(side=tk.RIGHT, padx=10)

        # White status row
        self.white_row = tk.Frame(score_frame, bg="#1E1E2A")
        self.white_row.pack(fill=tk.X, pady=4)
        tk.Label(self.white_row, text="○ 백 (White)", font=("Segoe UI", 11, "bold"), bg="#1E1E2A", fg="#E0E0FF").pack(side=tk.LEFT)
        self.white_time_lbl = tk.Label(self.white_row, text="60.0s", font=("Consolas", 12, "bold"), bg="#1E1E2A", fg="#34C759")
        self.white_time_lbl.pack(side=tk.RIGHT)
        self.white_cap_lbl = tk.Label(self.white_row, text="따냄: 0", font=("Segoe UI", 9), bg="#1E1E2A", fg=self.TEXT_MUTED)
        self.white_cap_lbl.pack(side=tk.RIGHT, padx=10)

        # Active Turn Banner
        self.turn_banner = tk.Label(
            side_panel,
            text="대국 시작: 흑의 차례입니다",
            font=("Segoe UI", 10, "bold"),
            bg="#3B3B55",
            fg="#FFFFFF",
            pady=8,
            relief=tk.FLAT
        )
        self.turn_banner.pack(fill=tk.X, pady=(0, 14))

        # Settings Section
        settings_box = tk.LabelFrame(
            side_panel,
            text=" 대국 설정 ",
            font=("Segoe UI", 9, "bold"),
            bg=self.PANEL_BG,
            fg=self.TEXT_LIGHT,
            padx=10,
            pady=8
        )
        settings_box.pack(fill=tk.X, pady=(0, 14))

        # Opponent Selector
        tk.Label(settings_box, text="AI 상대 선택:", font=("Segoe UI", 9), bg=self.PANEL_BG, fg=self.TEXT_LIGHT).pack(anchor="w")
        self.ai_combo = ttk.Combobox(
            settings_box,
            state="readonly",
            values=[
                "SmartPlayer (AlphaZero 2x Ensemble - 최강)",
                "RLMCTSPlayer (순수 AlphaZero 2x MCTS)",
                "HeuristicSearchPlayer (미니맥스 탐색봇)",
                "SimplePlayer (경량 룰봇)",
                "Human vs Human (자유 2인 모드)"
            ],
            font=("Segoe UI", 9)
        )
        self.ai_combo.current(0)
        self.ai_combo.pack(fill=tk.X, pady=(2, 8))
        self.ai_combo.bind("<<ComboboxSelected>>", self._on_ai_mode_changed)

        # Player Color Selector
        tk.Label(settings_box, text="내 돌 색상 선택:", font=("Segoe UI", 9), bg=self.PANEL_BG, fg=self.TEXT_LIGHT).pack(anchor="w")
        color_frame = tk.Frame(settings_box, bg=self.PANEL_BG)
        color_frame.pack(fill=tk.X, pady=(2, 4))
        self.color_var = tk.IntVar(value=Board.BLACK)
        rb_black = tk.Radiobutton(
            color_frame,
            text="흑 (선공)",
            variable=self.color_var,
            value=Board.BLACK,
            bg=self.PANEL_BG,
            fg="#FFFFFF",
            selectcolor=self.BG_DARK,
            activebackground=self.PANEL_BG,
            activeforeground="#FFFFFF",
            command=self._on_color_changed
        )
        rb_black.pack(side=tk.LEFT, padx=(0, 10))
        rb_white = tk.Radiobutton(
            color_frame,
            text="백 (후공)",
            variable=self.color_var,
            value=Board.WHITE,
            bg=self.PANEL_BG,
            fg="#FFFFFF",
            selectcolor=self.BG_DARK,
            activebackground=self.PANEL_BG,
            activeforeground="#FFFFFF",
            command=self._on_color_changed
        )
        rb_white.pack(side=tk.LEFT)

        # Control Buttons
        btn_frame = tk.Frame(side_panel, bg=self.PANEL_BG)
        btn_frame.pack(fill=tk.X, pady=(0, 14))

        self.btn_new = tk.Button(
            btn_frame,
            text="🔄 새 게임 (New Game)",
            font=("Segoe UI", 9, "bold"),
            bg="#2E7D32",
            fg="#FFFFFF",
            activebackground="#1B5E20",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            pady=6,
            command=self.reset_game
        )
        self.btn_new.pack(fill=tk.X, pady=(0, 6))

        action_row = tk.Frame(btn_frame, bg=self.PANEL_BG)
        action_row.pack(fill=tk.X)

        self.btn_undo = tk.Button(
            action_row,
            text="↩️ 한 수 물리기",
            font=("Segoe UI", 9),
            bg="#455A64",
            fg="#FFFFFF",
            relief=tk.FLAT,
            pady=4,
            command=self.undo_move
        )
        self.btn_undo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self.btn_hint = tk.Button(
            action_row,
            text="💡 AI 힌트",
            font=("Segoe UI", 9),
            bg="#0288D1",
            fg="#FFFFFF",
            relief=tk.FLAT,
            pady=4,
            command=self.request_hint
        )
        self.btn_hint.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        self.btn_rules = tk.Button(
            btn_frame,
            text="📖 Omok++ 대회 규칙 안내",
            font=("Segoe UI", 9),
            bg="#4E342E",
            fg="#FFE082",
            relief=tk.FLAT,
            pady=4,
            command=self.show_rules
        )
        self.btn_rules.pack(fill=tk.X, pady=(6, 0))

        # Move History Section
        history_box = tk.LabelFrame(
            side_panel,
            text=" 수순 기록 (Move Log) ",
            font=("Segoe UI", 9, "bold"),
            bg=self.PANEL_BG,
            fg=self.TEXT_LIGHT,
            padx=6,
            pady=4
        )
        history_box.pack(fill=tk.BOTH, expand=True)

        self.history_text = tk.Text(
            history_box,
            font=("Consolas", 9),
            bg="#181824",
            fg="#C0C0D0",
            relief=tk.FLAT,
            height=6,
            wrap=tk.WORD
        )
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(history_box, command=self.history_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text.config(yscrollcommand=scroll.set)

    def _init_ai_player(self):
        """Initializes selected AI player in background or on demand."""
        mode = self.ai_combo.get()
        self.ai_mode = mode

        if "Human vs Human" in mode:
            self.ai_instance = None
            return

        try:
            if "SmartPlayer" in mode:
                from players.smart_player import SmartPlayer
                self.ai_instance = SmartPlayer(self.ai_player_id)
            elif "RLMCTSPlayer" in mode:
                from players.rl_mcts_player import RLMCTSPlayer
                self.ai_instance = RLMCTSPlayer(
                    self.ai_player_id,
                    checkpoint_path=os.path.join(PROJECT_ROOT, "models", "checkpoints", "rl_model_best.pt"),
                    default_sims=50
                )
            elif "HeuristicSearchPlayer" in mode:
                from players.heuristic_search_player import HeuristicSearchPlayer
                self.ai_instance = HeuristicSearchPlayer(self.ai_player_id, depth=2)
            elif "SimplePlayer" in mode:
                from players.simpleplayer import SimplePlayer
                self.ai_instance = SimplePlayer(self.ai_player_id)
        except Exception as e:
            print(f"Failed to initialize {mode}: {e}")
            from players.simpleplayer import SimplePlayer
            self.ai_instance = SimplePlayer(self.ai_player_id)

    def _on_ai_mode_changed(self, event=None):
        self._init_ai_player()
        if not self.game_over and self.current_player == self.ai_player_id and self.ai_instance:
            self._trigger_ai_move()

    def _on_color_changed(self):
        new_human_color = self.color_var.get()
        if new_human_color != self.human_player:
            self.human_player = new_human_color
            self.ai_player_id = 1 - self.human_player
            self.reset_game()

    def reset_game(self):
        self.board = Board()
        self.current_player = Board.BLACK
        self.game_over = False
        self.winner = None
        self.history.clear()
        self.time_left = {Board.BLACK: self.TOTAL_TIME, Board.WHITE: self.TOTAL_TIME}
        self.timer_active = True
        self.last_timer_tick = time.time()

        self.history_text.delete("1.0", tk.END)
        self._init_ai_player()
        self._update_scoreboard()
        self._redraw_board()

        self.turn_banner.config(
            text="새 대국: 흑(선공)의 차례입니다",
            bg="#3B3B55",
            fg="#FFFFFF"
        )

        # If human chose White, AI (Black) moves first!
        if self.human_player == Board.WHITE and self.ai_instance:
            self._trigger_ai_move()

    def _timer_tick(self):
        now = time.time()
        if self.timer_active and not self.game_over:
            if self.last_timer_tick > 0:
                elapsed = now - self.last_timer_tick
                self.time_left[self.current_player] = max(0.0, self.time_left[self.current_player] - elapsed)
                if self.time_left[self.current_player] <= 0.0:
                    self._handle_timeout(self.current_player)

        self.last_timer_tick = now
        self._update_scoreboard()
        self.root.after(100, self._timer_tick)

    def _handle_timeout(self, timed_out_player: int):
        self.game_over = True
        self.winner = 1 - timed_out_player
        winner_name = "백 (White)" if self.winner == Board.WHITE else "흑 (Black)"
        loser_name = "흑 (Black)" if timed_out_player == Board.BLACK else "백 (White)"

        self.turn_banner.config(
            text=f"⏱️ 시간 초과! {loser_name} 패배 | {winner_name} 승리!",
            bg="#B71C1C",
            fg="#FFFFFF"
        )
        messagebox.showinfo("시간 초과", f"{loser_name}의 총 대국 시간 60초가 만료되었습니다.\n{winner_name}의 시간승!")

    def _coord_to_rc(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Converts pixel coordinate to nearest board grid row, col."""
        c = round((x - self.margin) / self.cell_size)
        r = round((y - self.margin) / self.cell_size)
        if 0 <= r < 19 and 0 <= c < 19:
            # Check Euclidean distance threshold (within 16px of intersection)
            px = self.margin + c * self.cell_size
            py = self.margin + r * self.cell_size
            if math.hypot(x - px, y - py) <= self.cell_size * 0.48:
                return (r, c)
        return None

    def _rc_to_coord(self, r: int, c: int) -> Tuple[float, float]:
        """Converts grid (r, c) to pixel (x, y)."""
        x = self.margin + c * self.cell_size
        y = self.margin + r * self.cell_size
        return (x, y)

    def _on_canvas_click(self, event):
        if self.game_over or self.ai_thinking:
            return

        # Check if it's human's turn (or 2-player mode)
        is_two_player = "Human vs Human" in self.ai_combo.get()
        if not is_two_player and self.current_player != self.human_player:
            return

        rc = self._coord_to_rc(event.x, event.y)
        if rc is None:
            return

        r, c = rc
        if not self.board.is_empty(r, c):
            return

        self._execute_move(r, c)

    def _execute_move(self, r: int, c: int):
        """Places stone, updates game, triggers captures and win checks."""
        self.history.append(self.board.copy())

        mover = self.current_player
        captured = self.board.place_stone(r, c, mover)

        # Record in move log
        move_num = len(self.board.move_history)
        col_str = self.COL_NAMES[c]
        row_str = str(19 - r)
        player_str = "흑" if mover == Board.BLACK else "백"
        cap_str = f" [+{len(captured)}포획]" if captured else ""
        log_entry = f"{move_num:3d}. {player_str} {col_str}{row_str}{cap_str}\n"
        self.history_text.insert(tk.END, log_entry)
        self.history_text.see(tk.END)

        # Check terminal state
        is_over, winner = self.board.check_game_status()

        self._redraw_board(captured_highlight=captured)
        self._update_scoreboard()

        if is_over:
            self._handle_game_over(winner)
            return

        # Switch turn
        self.current_player = 1 - self.current_player

        # Check if AI should move
        is_two_player = "Human vs Human" in self.ai_combo.get()
        if not is_two_player and self.current_player == self.ai_player_id:
            self._trigger_ai_move()
        else:
            turn_name = "흑(선공)" if self.current_player == Board.BLACK else "백(후공)"
            self.turn_banner.config(
                text=f"당신의 차례입니다 ({turn_name})",
                bg="#3B3B55",
                fg="#FFFFFF"
            )

    def _trigger_ai_move(self):
        """Launches AI computation in a background thread."""
        self.ai_thinking = True
        self.canvas.config(cursor="watch")
        ai_name = "백" if self.ai_player_id == Board.WHITE else "흑"
        self.turn_banner.config(
            text=f"🤖 AI ({ai_name}) 수읽기 중... (MCTS 시뮬레이션)",
            bg="#D97706",
            fg="#FFFFFF"
        )

        def worker():
            start_t = time.time()
            time_left_ms = self.time_left[self.ai_player_id] * 1000.0
            grid_copy = self.board.get_grid()

            try:
                move = self.ai_instance.take_turn(grid_copy, time_left_ms)
            except Exception as e:
                print(f"AI computation error: {e}")
                # Fallback to legal move
                legal = self.board.get_legal_moves()
                move = legal[0] if legal else (9, 9)

            elapsed = time.time() - start_t
            # Minimum brief delay so moves don't appear instantaneously erratic
            if elapsed < 0.2:
                time.sleep(0.2 - elapsed)

            self.root.after(0, self._apply_ai_move, move)

        threading.Thread(target=worker, daemon=True).start()

    def _apply_ai_move(self, move: Tuple[int, int]):
        self.ai_thinking = False
        self.canvas.config(cursor="crosshair")
        if self.game_over:
            return
        r, c = int(move[0]), int(move[1])
        self._execute_move(r, c)

    def _handle_game_over(self, winner: Optional[int]):
        self.game_over = True
        self.timer_active = False

        if winner == -1:
            msg = "🤝 무승부! (동시 5목 달성 또는 판 가득 참)"
            self.turn_banner.config(text=msg, bg="#4B5563", fg="#FFFFFF")
            messagebox.showinfo("대국 종료", msg)
        elif winner == Board.BLACK:
            msg = "🏆 흑 (Black) 승리! — 정확한 5목 달성"
            self.turn_banner.config(text=msg, bg="#166534", fg="#FFFFFF")
            messagebox.showinfo("승리", msg)
        elif winner == Board.WHITE:
            msg = "🏆 백 (White) 승리! — 정확한 5목 달성"
            self.turn_banner.config(text=msg, bg="#1E40AF", fg="#FFFFFF")
            messagebox.showinfo("승리", msg)

    def undo_move(self):
        if self.ai_thinking or not self.history:
            return

        is_two_player = "Human vs Human" in self.ai_combo.get()
        if is_two_player:
            # Undo 1 move
            self.board = self.history.pop()
            self.current_player = 1 - self.current_player
        else:
            # Undo 2 moves (AI + Human) to restore Human's turn
            if len(self.history) >= 2:
                self.history.pop()
                self.board = self.history.pop()
            elif len(self.history) == 1:
                self.board = self.history.pop()
                self.current_player = Board.BLACK

        self.game_over = False
        self.winner = None
        self.timer_active = True
        self._redraw_board()
        self._update_scoreboard()
        self.turn_banner.config(text="한 수 물리기 완료. 당신의 차례입니다.", bg="#3B3B55", fg="#FFFFFF")

    def request_hint(self):
        """Consults the AI for a recommended move."""
        if self.game_over or self.ai_thinking:
            return

        try:
            from players.smart_player import SmartPlayer
            hint_bot = SmartPlayer(self.current_player)
            time_ms = 5000.0
            move = hint_bot.take_turn(self.board.get_grid(), time_ms)
            r, c = move
            col_str = self.COL_NAMES[c]
            row_str = str(19 - r)
            self._redraw_board(hint_coord=(r, c))
            messagebox.showinfo("AI 추천수 힌트", f"AI 추천 착수 위치: {col_str}{row_str} (행: {r}, 열: {c})")
        except Exception as e:
            messagebox.showerror("힌트 오류", f"힌트 생성 실패: {e}")

    def show_rules(self):
        rules_text = (
            "【 Omok++ 대회 공식 규정 】\n\n"
            "1. 바둑판 규격: 19×19 정규 격자\n"
            "   - 흑(선공, 0), 백(후공, 1)\n\n"
            "2. 승리 조건: 정확한 5목 (Exact 5-in-a-Row)\n"
            "   - 가로, 세로, 대각선 중 '정확히 5개의 연속된 돌'이 놓여야 승리합니다.\n"
            "   - ★ 장목(6목 이상)은 승리로 인정되지 않습니다!\n"
            "   - 동시 5목 달성 시 무승부(Tie) 처리됩니다.\n\n"
            "3. 포획(따내기, Custodial Capture) 규정:\n"
            "   - 상대방의 돌 '정확히 2개'가 일직선으로 붙어있을 때,\n"
            "     양 끝을 내 돌로 감싸면(내돌-상대-상대-내돌) 그 상대 돌 2개를 따냅니다!\n"
            "   - 가로, 세로, 대각선 8방향 모두 적용되며, 한 수로 여러 방향 동시 포획도 가능합니다.\n"
            "   - ★ 반전 트릭: 상대의 6목 중 1개를 따내서 5목으로 줄여버리면 상대가 즉시 승리할 수도 있습니다!\n\n"
            "4. 시간 규정:\n"
            "   - 각 플레이어당 총 60.0초(턴당이 아님)의 제한 시간이 주어집니다.\n"
            "   - 시간 초과 시 즉시 시간패(Time Forfeit)합니다."
        )
        messagebox.showinfo("Omok++ 공식 규칙", rules_text)

    def _update_scoreboard(self):
        # Update captures
        self.black_cap_lbl.config(text=f"따냄: {self.board.captures[Board.BLACK]}")
        self.white_cap_lbl.config(text=f"따냄: {self.board.captures[Board.WHITE]}")

        # Update timers
        b_time = self.time_left[Board.BLACK]
        w_time = self.time_left[Board.WHITE]
        self.black_time_lbl.config(text=f"{b_time:4.1f}s", fg="#FF3B30" if b_time < 10.0 else "#34C759")
        self.white_time_lbl.config(text=f"{w_time:4.1f}s", fg="#FF3B30" if w_time < 10.0 else "#34C759")

    def _redraw_board(self, hint_coord=None, captured_highlight=None):
        self.canvas.delete("all")

        # 1. Draw Wooden Background Gradient
        self.canvas.create_rectangle(
            0, 0, self.board_pixel_size, self.board_pixel_size,
            fill=self.BOARD_WOOD, outline=""
        )

        # Subtle wood border line
        self.canvas.create_rectangle(
            self.margin - 12, self.margin - 12,
            self.board_pixel_size - self.margin + 12,
            self.board_pixel_size - self.margin + 12,
            outline="#7C5824", width=3
        )

        # 2. Draw Coordinates (A-S, 19-1)
        for i in range(19):
            x = self.margin + i * self.cell_size
            y = self.margin + i * self.cell_size
            col_letter = self.COL_NAMES[i]
            row_num = str(19 - i)

            # Top & Bottom Col labels
            self.canvas.create_text(x, self.margin - 24, text=col_letter, fill=self.GRID_LINE, font=("Segoe UI", 9, "bold"))
            self.canvas.create_text(x, self.board_pixel_size - self.margin + 24, text=col_letter, fill=self.GRID_LINE, font=("Segoe UI", 9, "bold"))

            # Left & Right Row labels
            self.canvas.create_text(self.margin - 24, y, text=row_num, fill=self.GRID_LINE, font=("Segoe UI", 9, "bold"))
            self.canvas.create_text(self.board_pixel_size - self.margin + 24, y, text=row_num, fill=self.GRID_LINE, font=("Segoe UI", 9, "bold"))

        # 3. Draw Grid Lines
        for i in range(19):
            coord = self.margin + i * self.cell_size
            # Horizontal line
            self.canvas.create_line(
                self.margin, coord,
                self.board_pixel_size - self.margin, coord,
                fill=self.GRID_LINE, width=1.2
            )
            # Vertical line
            self.canvas.create_line(
                coord, self.margin,
                coord, self.board_pixel_size - self.margin,
                fill=self.GRID_LINE, width=1.2
            )

        # 4. Draw Star Points (화점)
        for r, c in self.STAR_POINTS:
            x, y = self._rc_to_coord(r, c)
            self.canvas.create_oval(x - 3.5, y - 3.5, x + 3.5, y + 3.5, fill=self.STAR_POINT, outline="")

        # 5. Draw Captured Highlight (if any)
        if captured_highlight:
            for cr, cc in captured_highlight:
                cx, cy = self._rc_to_coord(cr, cc)
                self.canvas.create_oval(
                    cx - 14, cy - 14, cx + 14, cy + 14,
                    outline="#FF1744", width=3, dash=(4, 2)
                )

        # 6. Draw Stones
        r_radius = self.cell_size * 0.44
        grid = self.board.grid
        for r in range(19):
            for c in range(19):
                stone = grid[r][c]
                if stone == Board.EMPTY:
                    continue
                x, y = self._rc_to_coord(r, c)
                self._draw_stone(x, y, stone, r_radius)

        # 7. Highlight Last Move
        if self.board.last_move:
            lr, lc = self.board.last_move
            lx, ly = self._rc_to_coord(lr, lc)
            # Draw distinctive inner marker
            self.canvas.create_oval(
                lx - 4, ly - 4, lx + 4, ly + 4,
                fill=self.HIGHLIGHT_RED, outline="#FFFFFF", width=1.2
            )

        # 8. Draw Hint Coordinate (if requested)
        if hint_coord:
            hr, hc = hint_coord
            hx, hy = self._rc_to_coord(hr, hc)
            self.canvas.create_oval(
                hx - r_radius - 2, hy - r_radius - 2,
                hx + r_radius + 2, hy + r_radius + 2,
                outline=self.HIGHLIGHT_GOLD, width=3
            )
            self.canvas.create_text(
                hx, hy, text="★", fill=self.HIGHLIGHT_GOLD, font=("Segoe UI", 12, "bold")
            )

    def _draw_stone(self, x: float, y: float, player: int, radius: float):
        """Renders 3D realistic stone with specular lighting."""
        if player == Board.BLACK:
            # Shadow
            self.canvas.create_oval(
                x - radius + 1.5, y - radius + 2.5,
                x + radius + 1.5, y + radius + 2.5,
                fill="#1A140E", outline=""
            )
            # Main black body
            self.canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                fill="#1C1C1E", outline="#000000", width=1
            )
            # Specular reflection highlight
            spec_r = radius * 0.35
            self.canvas.create_oval(
                x - radius * 0.45, y - radius * 0.45,
                x - radius * 0.45 + spec_r, y - radius * 0.45 + spec_r,
                fill="#50505A", outline=""
            )
        else:
            # Shadow
            self.canvas.create_oval(
                x - radius + 1.5, y - radius + 2.5,
                x + radius + 1.5, y + radius + 2.5,
                fill="#8C7355", outline=""
            )
            # Main white body
            self.canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                fill="#F8F8FC", outline="#D0D0DC", width=1
            )
            # Specular highlight
            spec_r = radius * 0.35
            self.canvas.create_oval(
                x - radius * 0.45, y - radius * 0.45,
                x - radius * 0.45 + spec_r, y - radius * 0.45 + spec_r,
                fill="#FFFFFF", outline=""
            )

    def _on_canvas_motion(self, event):
        """Shows subtle hover shadow on intersection."""
        if self.game_over or self.ai_thinking:
            return
        is_two_player = "Human vs Human" in self.ai_combo.get()
        if not is_two_player and self.current_player != self.human_player:
            return

        rc = self._coord_to_rc(event.x, event.y)
        self.canvas.delete("hover_ghost")

        if rc and self.board.is_empty(rc[0], rc[1]):
            x, y = self._rc_to_coord(rc[0], rc[1])
            radius = self.cell_size * 0.40
            outline_color = "#333333" if self.current_player == Board.BLACK else "#EEEEEE"
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline=outline_color, width=2, dash=(3, 3), tags="hover_ghost"
            )

    def _on_canvas_leave(self, event):
        self.canvas.delete("hover_ghost")


def launch():
    root = tk.Tk()
    app = OmokGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
