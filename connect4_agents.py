import random
from typing import List
from Connect4Env import Connect4Env
from connect4_bot import Connect4Bot

class RandomAgent:
    def get_move(self, env: Connect4Env, current_player: int) -> int:
        bot = Connect4Bot(max_depth=6)
        return bot.get_move(env=env,current_player=current_player) 
    