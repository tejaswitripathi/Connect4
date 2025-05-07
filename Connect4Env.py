import numpy as np
import random
from typing import List

class Connect4Env:
    def __init__(self, rows: int = 6, cols: int = 7, initial_blocked: int = 4):
        self.rows = rows
        self.cols = cols
        # 0: empty, 1: yellow, 2: red
        self.board = np.zeros((rows, cols), dtype=int)
        # mask of currently blocked cells
        self.blocked = np.zeros((rows, cols), dtype=bool)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        # first set of blocked cells
        self._update_blocked_cells(initial_blocked)
        # Store previous state for undo
        self._prev_board = None
        self._prev_blocked = None
        self._prev_player = None
        self._prev_game_over = None
        self._prev_winner = None

    def _update_blocked_cells(self, num_blocked: int = None):
        """Randomize 4 blocked positions; remove any discs in them."""
        # clear old blocks
        self.blocked[:, :] = False

        # choose how many to block
        n = num_blocked if num_blocked is not None else random.choice([4, 5])
        all_positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        for r, c in random.sample(all_positions, n):
            self.blocked[r, c] = True
            # if there was a disc here, remove it
            if self.board[r, c] in (1, 2):
                self.board[r, c] = 0

    def make_move(self, col: int) -> bool:
        if self.game_over:
            return False

        # Store current state before making move
        self._prev_board = self.board.copy()
        self._prev_blocked = self.blocked.copy()
        self._prev_player = self.current_player
        self._prev_game_over = self.game_over
        self._prev_winner = self.winner

        # on each turn, pick 4–5 new blocked cells
        self._update_blocked_cells()

        # drop in lowest non-blocked empty cell
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0 and not self.blocked[row, col]:
                self.board[row, col] = self.current_player

                # check for win
                if self._check_win(row, col):
                    self.game_over = True
                    self.winner = self.current_player
                    return True

                # check for draw
                if np.all((self.board != 0) | self.blocked):
                    self.game_over = True
                    self.winner = 0
                    return True

                # switch
                self.current_player = 3 - self.current_player
                return True

        # column full or completely blocked
        return False

    def undo_move(self) -> None:
        """Undo the last move and restore the previous state."""
        if self._prev_board is not None:
            self.board = self._prev_board
            self.blocked = self._prev_blocked
            self.current_player = self._prev_player
            self.game_over = self._prev_game_over
            self.winner = self._prev_winner
            # Clear previous state
            self._prev_board = None
            self._prev_blocked = None
            self._prev_player = None
            self._prev_game_over = None
            self._prev_winner = None

    def _check_win(self, row: int, col: int) -> bool:
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # forward
            r, c = row + dr, col + dc
            while (0 <= r < self.rows and 0 <= c < self.cols and
                   self.board[r, c] == player and not self.blocked[r, c]):
                count += 1
                r += dr; c += dc
            # backward
            r, c = row - dr, col - dc
            while (0 <= r < self.rows and 0 <= c < self.cols and
                   self.board[r, c] == player and not self.blocked[r, c]):
                count += 1
                r -= dr; c -= dc
            if count >= 4:
                return True
        return False

    def get_valid_moves(self) -> List[int]:
        """Columns where the top cell is empty and not blocked."""
        return [
            c for c in range(self.cols)
            if self.board[0, c] == 0 and not self.blocked[0, c]
        ]

    def display(self):
        symbols = {0: '⚪', 1: '🟡', 2: '🔴'}
        print('   ' + '   '.join(map(str, range(self.cols))))
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if self.blocked[r, c]:
                    row_str.append('⬛')
                else:
                    row_str.append(symbols[self.board[r, c]])
            print(f'{r}  ' + ' '.join(row_str))
        print()

    def copy(self):
        env = Connect4Env(self.rows, self.cols, 0)
        env.board = self.board.copy()
        env.blocked = self.blocked.copy()
        env.current_player = self.current_player
        env.game_over = self.game_over
        env.winner = self.winner
        return env