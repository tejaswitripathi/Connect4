import numpy as np
from typing import Tuple, List, Optional
from Connect4Env import Connect4Env

class Connect4Bot:
    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth
        self.center_weight = 2  # Reduced from 3 since center control is less important with blocking
        self.blocking_penalty = 1.5  # Penalty for moves that could be blocked
        self.pattern_weights = {
            4: 100,  # 4 in a row
            3: 5,    # 3 in a row
            2: 2     # 2 in a row
        }
        self._move_cache = {}  # Cache for move evaluations

    def get_move(self, env: Connect4Env, current_player: int) -> int:
        """
        Get the best move for the current player using minimax with alpha-beta pruning.
        Checks for immediate win/block opportunities before deeper search.
        Returns the column index of the best move.
        """
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return -1

        # Check for immediate win
        for move in valid_moves:
            if self._is_winning_move(env, move, current_player):
                return move

        # Check for opponent's immediate win
        opponent = 3 - current_player
        for move in valid_moves:
            if self._is_winning_move(env, move, opponent):
                return move

        # Sort moves by center preference and blocking risk
        center = env.cols // 2
        valid_moves.sort(key=lambda c: (abs(c - center), -self._get_column_blocking_risk(env, c)))

        best_score = float('-inf')
        best_move = valid_moves[0]
        alpha = float('-inf')
        beta = float('inf')

        # Clear cache at start of new move
        self._move_cache.clear()

        for move in valid_moves:
            # Make move and evaluate
            env.make_move(move)
            score = self._minimax(env, self.max_depth - 1, False, current_player, alpha, beta)
            env.undo_move()  # Undo the move

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move

    def _is_winning_move(self, env: Connect4Env, move: int, player: int) -> bool:
        """Check if a move would result in an immediate win"""
        env.make_move(move)
        is_win = env.game_over and env.winner == player
        env.undo_move()
        return is_win

    def _calculate_blocking_probabilities(self, env: Connect4Env) -> np.ndarray:
        """Calculate blocking probabilities for each cell based on Manhattan distance."""
        blocking_probs = np.zeros((env.rows, env.cols))
        for r in range(env.rows):
            for c in range(env.cols):
                if env.board[r, c] == 0 and not env.blocked[r, c]:
                    min_dist = float('inf')
                    for r2 in range(env.rows):
                        for c2 in range(env.cols):
                            if env.board[r2, c2] in (1, 2):
                                dist = abs(r - r2) + abs(c - c2)
                                min_dist = min(min_dist, dist)
                    if min_dist != float('inf'):
                        blocking_probs[r, c] = 1 / (1 + np.exp(-0.1 * min_dist))
        return blocking_probs

    def _get_column_blocking_risk(self, env: Connect4Env, col: int) -> float:
        """Calculate how likely a column is to be blocked based on current pattern and blocking probabilities."""
        risk = 0
        blocking_probs = self._calculate_blocking_probabilities(env)
        
        for row in range(env.rows):
            if env.board[row, col] == 0 and not env.blocked[row, col]:
                # Add risk based on blocking probability
                risk += blocking_probs[row, col] * self.blocking_penalty
                
                # Check if this cell is part of any potential winning patterns
                for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    count = 1
                    # Check in both directions
                    for direction in [1, -1]:
                        r, c = row + dr * direction, col + dc * direction
                        while (0 <= r < env.rows and 0 <= c < env.cols and
                               env.board[r, c] != 0 and not env.blocked[r, c]):
                            count += 1
                            r += dr * direction
                            c += dc * direction
                    if count >= 3:  # If there's a potential 3-in-a-row
                        risk += self.blocking_penalty
        return risk

    def _minimax(self, env: Connect4Env, depth: int, is_maximizing: bool,
                 current_player: int, alpha: float, beta: float) -> float:
        """
        Minimax with alpha-beta pruning, adapted for dynamic blocking.
        """
        # Check cache first
        cache_key = (tuple(map(tuple, env.board)), tuple(map(tuple, env.blocked)), depth, is_maximizing)
        if cache_key in self._move_cache:
            return self._move_cache[cache_key]

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
                env.make_move(move)
                eval = self._minimax(env, depth - 1, False, current_player, alpha, beta)
                env.undo_move()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self._move_cache[cache_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - current_player
            for move in valid_moves:
                env.make_move(move)
                eval = self._minimax(env, depth - 1, True, current_player, alpha, beta)
                env.undo_move()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self._move_cache[cache_key] = min_eval
            return min_eval

    def _evaluate_position(self, env: Connect4Env, current_player: int) -> float:
        score = 0
        board = env.board
        rows, cols = env.rows, env.cols
        opponent = 3 - current_player
        blocking_probs = self._calculate_blocking_probabilities(env)

        # Score center column (reduced weight due to blocking)
        center_array = list(board[:, cols // 2])
        score += center_array.count(current_player) * self.center_weight
        score -= center_array.count(opponent) * self.center_weight

        # Define helper to score a window of 4, considering blocked cells and blocking probabilities
        def score_window(window: List[int], blocked: List[bool], window_probs: List[float]) -> float:
            s = 0
            # Count pieces and empty spaces, ignoring blocked cells
            player_count = sum(1 for i, val in enumerate(window) if val == current_player and not blocked[i])
            opp_count = sum(1 for i, val in enumerate(window) if val == opponent and not blocked[i])
            empty_count = sum(1 for i, val in enumerate(window) if val == 0 and not blocked[i])
            
            # Calculate blocking risk for this window
            blocking_risk = sum(window_probs)
            
            if player_count == 4:
                s += self.pattern_weights[4]
            elif player_count == 3 and empty_count == 1:
                s += self.pattern_weights[3] * (1 - blocking_risk)
            elif player_count == 2 and empty_count == 2:
                s += self.pattern_weights[2] * (1 - blocking_risk)
                
            if opp_count == 3 and empty_count == 1:
                s -= self.pattern_weights[3] * 0.8 * (1 - blocking_risk)
            return s

        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                window = list(board[r, c:c + 4])
                blocked = list(env.blocked[r, c:c + 4])
                window_probs = list(blocking_probs[r, c:c + 4])
                score += score_window(window, blocked, window_probs)

        # Vertical
        for c in range(cols):
            for r in range(rows - 3):
                window = list(board[r:r + 4, c])
                blocked = list(env.blocked[r:r + 4, c])
                window_probs = list(blocking_probs[r:r + 4, c])
                score += score_window(window, blocked, window_probs)

        # Positive diagonal
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = [board[r + i, c + i] for i in range(4)]
                blocked = [env.blocked[r + i, c + i] for i in range(4)]
                window_probs = [blocking_probs[r + i, c + i] for i in range(4)]
                score += score_window(window, blocked, window_probs)

        # Negative diagonal
        for r in range(3, rows):
            for c in range(cols - 3):
                window = [board[r - i, c + i] for i in range(4)]
                blocked = [env.blocked[r - i, c + i] for i in range(4)]
                window_probs = [blocking_probs[r - i, c + i] for i in range(4)]
                score += score_window(window, blocked, window_probs)

        return score
