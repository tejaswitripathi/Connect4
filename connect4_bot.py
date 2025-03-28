import numpy as np
from typing import Tuple, List, Optional
from Connect4Env import Connect4Env

class Connect4Bot:
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        
    def get_move(self, env: Connect4Env, current_player: int) -> int:
        """
        Get the best move for the current player using minimax with alpha-beta pruning.
        Returns the column index of the best move.
        """
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return -1
            
        best_score = float('-inf')
        best_move = valid_moves[0]
        alpha = float('-inf')
        beta = float('inf')
        
        for move in valid_moves:
            # Make a copy of the environment to simulate the move
            env_copy = Connect4Env(env.rows, env.cols)
            env_copy.board = env.board.copy()
            env_copy.make_move(move, current_player)
            
            # Evaluate the move
            score = self._minimax(env_copy, self.max_depth - 1, False, current_player, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_move = move
                
            alpha = max(alpha, best_score)
            
        return best_move
    
    def _minimax(self, env: Connect4Env, depth: int, is_maximizing: bool, 
                 current_player: int, alpha: float, beta: float) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        Returns the evaluation score for the current position.
        """
        # Terminal states
        if env.game_over:
            if env.winner == current_player:
                return float('inf')
            elif env.winner == 3 - current_player:
                return float('-inf')
            else:  # Draw
                return 0
                
        if depth == 0:
            return self._evaluate_position(env, current_player)
            
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return 0
            
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                env_copy = Connect4Env(env.rows, env.cols)
                env_copy.board = env.board.copy()
                env_copy.make_move(move, current_player)
                
                eval = self._minimax(env_copy, depth - 1, False, current_player, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                env_copy = Connect4Env(env.rows, env.cols)
                env_copy.board = env.board.copy()
                env_copy.make_move(move, 3 - current_player)
                
                eval = self._minimax(env_copy, depth - 1, True, current_player, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_position(self, env: Connect4Env, current_player: int) -> float:
        """
        Evaluate the current position.
        Returns a score that represents how good the position is for the current player.
        """
        score = 0
        
        # Check for potential wins
        for col in range(env.cols):
            for row in range(env.rows):
                if env.board[row, col] == 0:
                    # Simulate placing a piece
                    env.board[row, col] = current_player
                    if env._check_win(row, col):
                        score += 100
                    env.board[row, col] = 0
                    
                    # Check opponent's potential win
                    env.board[row, col] = 3 - current_player
                    if env._check_win(row, col):
                        score -= 90
                    env.board[row, col] = 0
        
        # Center control
        center_col = env.cols // 2
        for row in range(env.rows):
            if env.board[row, center_col] == current_player:
                score += 3
            elif env.board[row, center_col] == 3 - current_player:
                score -= 3
                
        # Height of pieces
        for col in range(env.cols):
            for row in range(env.rows):
                if env.board[row, col] == current_player:
                    score += (env.rows - row) * 2
                elif env.board[row, col] == 3 - current_player:
                    score -= (env.rows - row) * 2
                    
        return score 