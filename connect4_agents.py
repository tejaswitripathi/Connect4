import random
from typing import List
from Connect4Env import Connect4Env

class RandomAgent:
    def get_move(self, env: Connect4Env, current_player: int) -> int:
        """Pick a random valid move."""
        valid_moves = env.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else -1 