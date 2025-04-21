import numpy as np
from typing import Tuple, List, Optional
from Connect4Env import Connect4Env

class Connect4Bot:
    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth
        self.center_weight = 3

    def get_move(self, env: Connect4Env, current_player: int) -> int:
        """
        Get the best move for the current player using minimax with alpha-beta pruning.
        Checks for immediate win/block opportunities before deeper search.
        Returns the column index of the best move.
        """
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return -1

        # immediate win
        for move in valid_moves:
            env_copy = env.copy()
            env_copy.current_player = current_player
            if env_copy.make_move(move) and env_copy.game_over and env_copy.winner == current_player:
                return move

        # block opponent's immediate win
        opponent = 3 - current_player
        for move in valid_moves:
            env_copy = env.copy()
            env_copy.current_player = opponent
            if env_copy.make_move(move) and env_copy.game_over and env_copy.winner == opponent:
                return move

        # move ordering: prefer center columns
        center = env.cols // 2
        valid_moves.sort(key=lambda c: abs(c - center))

        best_score = float('-inf')
        best_move = valid_moves[0]
        alpha = float('-inf')
        beta = float('inf')

        for move in valid_moves:
            env_copy = env.copy()
            env_copy.current_player = current_player
            env_copy.make_move(move)

            score = self._minimax(env_copy, self.max_depth - 1, False, current_player, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move

    def _minimax(self, env: Connect4Env, depth: int, is_maximizing: bool,
                 current_player: int, alpha: float, beta: float) -> float:
        """
        Minimax with alpha-beta pruning.
        """
        if env.game_over:
            if env.winner == current_player:
                return float('inf')
            elif env.winner == 3 - current_player:
                return float('-inf')
            else:
                return 0

        if depth == 0:
            return self._evaluate_position(env, current_player)

        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return 0

        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                env_copy = env.copy()
                env_copy.current_player = current_player
                env_copy.make_move(move)
                eval = self._minimax(env_copy, depth - 1, False, current_player, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - current_player
            for move in valid_moves:
                env_copy = env.copy()
                env_copy.current_player = opponent
                env_copy.make_move(move)
                eval = self._minimax(env_copy, depth - 1, True, current_player, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate_position(self, env: Connect4Env, current_player: int) -> float:
        score = 0
        board = env.board
        rows, cols = env.rows, env.cols
        opponent = 3 - current_player

        # Score center column
        center_array = list(board[:, cols // 2])
        score += center_array.count(current_player) * self.center_weight
        score -= center_array.count(opponent) * self.center_weight

        # Define helper to score a window of 4
        def score_window(window: List[int]) -> float:
            s = 0
            if window.count(current_player) == 4:
                s += 100
            elif window.count(current_player) == 3 and window.count(0) == 1:
                s += 5
            elif window.count(current_player) == 2 and window.count(0) == 2:
                s += 2

            if window.count(opponent) == 3 and window.count(0) == 1:
                s -= 4
            return s

        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                window = list(board[r, c:c + 4])
                score += score_window(window)

        # Vertical
        for c in range(cols):
            for r in range(rows - 3):
                window = list(board[r:r + 4, c])
                score += score_window(window)

        # Positive diagonal
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = [board[r + i, c + i] for i in range(4)]
                score += score_window(window)

        # Negative diagonal
        for r in range(3, rows):
            for c in range(cols - 3):
                window = [board[r - i, c + i] for i in range(4)]
                score += score_window(window)

        return score
