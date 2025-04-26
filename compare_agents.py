from Connect4Env import Connect4Env
from connect4_bot import Connect4Bot
from connect4_agents import RandomAgent
import time
from typing import Tuple, List
import numpy as np

class AgentMatch:
    def __init__(self, agent1_name: str, agent1: object, agent2_name: str, agent2: object):
        self.agent1_name = agent1_name
        self.agent1 = agent1
        self.agent2_name = agent2_name
        self.agent2 = agent2
        self.agent1_wins = 0
        self.agent2_wins = 0
        self.draws = 0
        self.agent1_times: List[float] = []
        self.agent2_times: List[float] = []
        self.move_counts: List[int] = []  # track moves per game
    
    def play_game(self) -> Tuple[int, float, float, int]:
        """Play a single game and return (winner, time1, time2, moves)."""
        env = Connect4Env()
        current_player = 1
        agent1_time = 0.0
        agent2_time = 0.0
        move_count = 0
        
        while not env.game_over:
#             env.display()
            move_count += 1
            start_time = time.time()
            
            if current_player == 1:
                move = self.agent1.get_move(env, current_player)
                agent1_time += time.time() - start_time
            else:
                move = self.agent2.get_move(env, current_player)
                agent2_time += time.time() - start_time
            
            if move == -1:
                # invalid move ends game
                winner = 2 if current_player == 1 else 1
                return winner, agent1_time, agent2_time, move_count
            
            env.make_move(move)
            current_player = 3 - current_player
        
#         env.display()
        return env.winner, agent1_time, agent2_time, move_count
    
    def play_matches(self, num_matches: int):
        """Play multiple matches and track stats including move counts."""
        for i in range(num_matches):
            print(f"\nPlaying game {i+1}/{num_matches}")
            winner, t1, t2, moves = self.play_game()
            
            if winner == 1:
                self.agent1_wins += 1
                print(f"{self.agent1_name} wins!")
            elif winner == 2:
                self.agent2_wins += 1
                print(f"{self.agent2_name} wins!")
            else:
                self.draws += 1
                print("Draw!")

            self.agent1_times.append(t1)
            self.agent2_times.append(t2)
            self.move_counts.append(moves)
    
    def print_results(self):
        """Print match stats including average game length."""
        total = self.agent1_wins + self.agent2_wins + self.draws
        print(f"\nResults for {self.agent1_name} vs {self.agent2_name}:")
        print(f"Total games: {total}")
        print(f"{self.agent1_name} wins: {self.agent1_wins} ({self.agent1_wins/total*100:.1f}%)")
        print(f"{self.agent2_name} wins: {self.agent2_wins} ({self.agent2_wins/total*100:.1f}%)")
        print(f"Draws: {self.draws} ({self.draws/total*100:.1f}%)")
        print(f"Average move time for {self.agent1_name}: {np.mean(self.agent1_times):.3f}s")
        print(f"Average move time for {self.agent2_name}: {np.mean(self.agent2_times):.3f}s")
        print(f"Average game length: {np.mean(self.move_counts):.1f} moves")

def main():
    minimax_bot = Connect4Bot(max_depth=6)
    random_bot = RandomAgent()

    match = AgentMatch("Minimax", minimax_bot, "Random", random_bot)
    print("\nPlaying Minimax vs Random...")
    match.play_matches(100)
    match.print_results()

if __name__ == "__main__":
    main()
