import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from Connect4Env import Connect4Env   # :contentReference[oaicite:0]{index=0}
from connect4_bot import Connect4Bot  
from connect4_agents import RandomAgent

def encode_board(board, blocked=None, token_hidden_prob=0.0):
    """
    Encode a Connect 4 board into a 4-channel tensor:
      - Channel 1: Player 1 tokens
      - Channel 2: Player 2 tokens
      - Channel 3: Empty cells
      - Channel 4: Blocked cells
    """
    channels = np.zeros((4, board.shape[0], board.shape[1]), dtype=np.float32)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            cell = board[i, j]
            if cell == 1:
                channels[0, i, j] = 1.0  # Player 1 token
            elif cell == 2:
                channels[1, i, j] = 1.0  # Player 2 token
            else:
                channels[2, i, j] = 1.0  # Empty cell
            if blocked is not None and blocked[i, j]:
                channels[3, i, j] = 1.0  # Blocked cell
    return channels

# Create the CNN 
class Connect4CNN(nn.Module):
    def __init__(self):
        super(Connect4CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)  # Changed to 4 input channels
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
      
        self.fc1 = nn.Linear(128 * 6 * 7, 128)
        self.fc2 = nn.Linear(128, 7) 

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  

def select_action(policy, state, env, device, token_hidden_prob=0.0, force_random=False):
    """
    Given a policy network and the current environment state, select an action.
    If force_random is True, select a random valid move regardless of the policy.
    """
    valid_moves = env.get_valid_moves()
    if not valid_moves:
        return -1, torch.tensor(0.0, device=device)
    
    if force_random:
        action = random.choice(valid_moves)
        return action, torch.tensor(0.0, device=device)
    
    state_encoded = encode_board(state, env.blocked, token_hidden_prob)
    state_tensor = torch.tensor(state_encoded, dtype=torch.float32).unsqueeze(0).to(device)
    logits = policy(state_tensor).squeeze(0)
    
    # Create a mask: for legal moves set to 0, illegal remain -inf
    mask = torch.full((7,), float('-inf')).to(device)
    for m in valid_moves:
        mask[m] = 0.0
    masked_logits = logits + mask
    probs = torch.softmax(masked_logits, dim=0)
    
    if torch.isnan(probs).any():
        return -1, torch.tensor(0.0, device=device)
    
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def train_rl(policy, optimizer, num_games=100, token_hidden_prob=0.3, device=torch.device("cpu"), eval_interval=1000):
    """
    Train the policy network using a REINFORCE approach.
    At the end of each game, a reward (+1 win, -1 loss, 0 draw) is used to update the policy.
    """
    policy.train()
    win_rates = []
    eval_games = []
    minimax_opponent = Connect4Bot(max_depth=4)  # Using a moderate depth for training
    
    for game in range(num_games):
        env = Connect4Env()
        log_probs = []  
        state = env.board.copy()
        
        while not env.game_over:
            if env.current_player == 2:  # Minimax agent's turn
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                action = minimax_opponent.get_move(env, env.current_player)
            else:  # RL agent's turn
                if not env.get_valid_moves():
                    break
                action, log_prob = select_action(policy, state, env, device, token_hidden_prob)
                
                if action == -1:
                    break
                log_probs.append(log_prob)
            
            env.make_move(action)
            state = env.board.copy()
        
        # Determine reward.
        if env.winner == 1:
            reward = 1.0
        elif env.winner == 2:
            reward = -1.0
        else:
            reward = 0.0
        
        game_loss = - sum(log_probs) * reward
        
        optimizer.zero_grad()
        game_loss.backward()
        optimizer.step()
        
        print(f"Game {game+1}/{num_games}, Reward: {reward}, Loss: {game_loss.item():.4f}")
        
        # Evaluate every eval_interval games
        if (game + 1) % eval_interval == 0:
            wins, losses, draws = evaluate_rl(policy, num_games=10, token_hidden_prob=token_hidden_prob, device=device, return_stats=True)
            win_rate = wins / (wins + losses + draws) * 100
            win_rates.append(win_rate)
            eval_games.append(game + 1)
            print(f"Evaluation at game {game+1}: Win Rate: {win_rate:.2f}%")
    
    return win_rates, eval_games

def evaluate_rl(policy, num_games=20, token_hidden_prob=0.3, device=torch.device("cpu"), return_stats=False, force_random=False):
    # Evaluate the agent
    policy.eval()
    wins, losses, draws = 0, 0, 0
    minimax_opponent = Connect4Bot(max_depth=4)  # Using same depth as training
    
    for game in range(num_games):
        env = Connect4Env()
        state = env.board.copy()
        
        while not env.game_over:
            if env.current_player == 2:  # Minimax agent's turn
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                action = minimax_opponent.get_move(env, env.current_player)
            else:  # RL agent's turn
                action, _ = select_action(policy, state, env, device, token_hidden_prob, force_random)
                if action == -1:
                    break
            
            env.make_move(action)
            state = env.board.copy()
        
        if env.winner == 1:
            wins += 1
        elif env.winner == 2:
            losses += 1
        else:
            draws += 1
    
    if return_stats:
        return wins, losses, draws
    
    total = wins + losses + draws
    print(f"Evaluation over {total} Game: Wins: {wins}, Losses: {losses}, Draws: {draws}, Win Rate: {wins/total*100:.2f}%")

def plot_training_progress(win_rates, eval_games):
    """
    Plot the win rates over training games.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(eval_games, win_rates, 'b-', label='Win Rate')
    plt.xlabel('Training Games')
    plt.ylabel('Win Rate (%)')
    plt.title('RL Agent Win Rate During Training')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_progress.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create and reset the policy network
    policy = Connect4CNN().to(device)
    # Reset all parameters
    for layer in policy.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    # Evaluate before any training with random moves
    print("\nPre-training evaluation with random moves (100 games):")
    evaluate_rl(policy, num_games=100, token_hidden_prob=0.3, device=device, force_random=True)
    
    # Evaluate before any training with untrained policy
    print("\nPre-training evaluation with untrained policy (100 games):")
    evaluate_rl(policy, num_games=100, token_hidden_prob=0.3, device=device)
    
    num_training_games = 10000
    print("\nStarting reinforcement learning training against Minimax opponent...")
    win_rates, eval_games = train_rl(policy, optimizer, num_games=num_training_games, token_hidden_prob=0.3, device=device, eval_interval=1000)
    
    print("\nPlotting training progress...")
    plot_training_progress(win_rates, eval_games)
    
    print("\nFinal evaluation of trained RL agent...")
    evaluate_rl(policy, num_games=100, token_hidden_prob=0.3, device=device)

if __name__ == "__main__":
    main()
