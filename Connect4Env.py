import numpy as np
import random
from typing import Tuple, List

class Connect4Env:
    def __init__(self, rows: int = 6, cols: int = 7, num_blocked: int = 5):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)  # 0: empty, 1: yellow, 2: red, 3: blocked
        self.current_player = 1  # 1: yellow, 2: red
        self.game_over = False
        self.winner = None
        
        # Randomly place blocked cells
        self._place_blocked_cells(num_blocked)
    
    def _place_blocked_cells(self, num_blocked: int):
        # Get all possible positions
        positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        # Randomly select positions to block
        blocked_positions = random.sample(positions, num_blocked)
        for r, c in blocked_positions:
            self.board[r, c] = 3
    
    def make_move(self, col: int) -> bool:
        if self.game_over:
            return False

        # Find the lowest empty cell in the column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                
                # Check for win
                if self._check_win(row, col):
                    self.game_over = True
                    self.winner = self.current_player
                    return True
                
                # Check for draw
                if np.all(self.board != 0):
                    self.game_over = True
                    self.winner = 0  # Draw
                    return True
                
                # Switch players
                self.current_player = 3 - self.current_player
                return True

        return False

    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move resulted in a win."""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # vertical, horizontal, diagonal down, diagonal up
        player = self.board[row, col]
        
        for dr, dc in directions:
            count = 1
            # Check in one direction
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check in opposite direction
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 4:
                return True
        
        return False
    
    def get_valid_moves(self) -> List[int]:
        """Return list of valid column indices."""
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def display(self):
        """Display the current state of the board."""
        symbols = {0: '⚪', 1: '🟡', 2: '🔴', 3: '⬛'}
        print('  ' + '   '.join(str(i) for i in range(self.cols)))
        for row in range(self.rows):
            print(f'{row} ' + ' '.join(symbols[self.board[row, col]] for col in range(self.cols)))
        print()
    
    def copy(self):
        """Create a deep copy of the environment."""
        env = Connect4Env(self.rows, self.cols, 0)  # Create with 0 blocked cells
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.game_over = self.game_over
        env.winner = self.winner
        return env 