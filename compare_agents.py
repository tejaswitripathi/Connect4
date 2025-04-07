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
        self.agent1_times = []
        self.agent2_times = []
    
    def play_game(self) -> Tuple[int, float, float]:
        """Play a single game and return winner and move times."""
        env = Connect4Env()
        current_player = 1
        agent1_time = 0
        agent2_time = 0
        
        while not env.game_over:
            env.display()  # Show the board state
            start_time = time.time()
            
            if current_player == 1:
                move = self.agent1.get_move(env, current_player)
                agent1_time += time.time() - start_time
            else:
                move = self.agent2.get_move(env, current_player)
                agent2_time += time.time() - start_time
            
            if move == -1:  # Invalid move
                if current_player == 1:
                    return 2, agent1_time, agent2_time  # Player 2 wins
                else:
                    return 1, agent1_time, agent2_time  # Player 1 wins
            
            env.make_move(move)
            current_player = 3 - current_player
        
        env.display()  # Show final board state
        return env.winner, agent1_time, agent2_time
    
    def play_matches(self, num_matches: int):
        """Play multiple matches and track statistics."""
        for game in range(num_matches):
            print(f"\nPlaying game {game + 1}/{num_matches}")
            winner, agent1_time, agent2_time = self.play_game()
            
            if winner == 1:
                self.agent1_wins += 1
                print(f"{self.agent1_name} wins!")
            elif winner == 2:
                self.agent2_wins += 1
                print(f"{self.agent2_name} wins!")
            else:
                self.draws += 1
                print("Draw!")
                
            self.agent1_times.append(agent1_time)
            self.agent2_times.append(agent2_time)
    
    def print_results(self):
        """Print the results of the matches."""
        total_games = self.agent1_wins + self.agent2_wins + self.draws
        print(f"\nResults for {self.agent1_name} vs {self.agent2_name}:")
        print(f"Total games: {total_games}")
        print(f"{self.agent1_name} wins: {self.agent1_wins} ({self.agent1_wins/total_games*100:.1f}%)")
        print(f"{self.agent2_name} wins: {self.agent2_wins} ({self.agent2_wins/total_games*100:.1f}%)")
        print(f"Draws: {self.draws} ({self.draws/total_games*100:.1f}%)")
        print(f"Average move time for {self.agent1_name}: {np.mean(self.agent1_times):.3f}s")
        print(f"Average move time for {self.agent2_name}: {np.mean(self.agent2_times):.3f}s")

def main():
    # Initialize agents
    minimax_bot = Connect4Bot(max_depth=6)
    random_bot = RandomAgent()
    
    # Play matches
    match = AgentMatch("Minimax", minimax_bot, "Random", random_bot)
    print("\nPlaying Minimax vs Random...")
    match.play_matches(10)  # Play 5 games
    match.print_results()

if __name__ == "__main__":
    main() 