import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Connect4Env import Connect4Env   # :contentReference[oaicite:0]{index=0}
from connect4_bot import Connect4Bot  

def encode_board(board, token_hidden_prob=0.3):
    """ Encode a Connect 4 board into a 3-channel tensor:
      - Channel 1: Player 1 tokens (visible if not hidden)
      - Channel 2: Player 2 tokens (visible if not hidden)
      - Channel 3: Hidden mask (1 if the cell is hidden, 0 otherwise)
      (no clue if this is right) """
    channels = np.zeros((3, board.shape[0], board.shape[1]), dtype=np.float32)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            cell = board[i, j]
            if cell == 1 or cell == 2:
                if random.random() < token_hidden_prob:
                    channels[2, i, j] = 1.0  # Hide token.
                else:
                    if cell == 1:
                        channels[0, i, j] = 1.0
                    elif cell == 2:
                        channels[1, i, j] = 1.0
    return channels

# Create the CNN 
class Connect4CNN(nn.Module):
    def __init__(self):
        super(Connect4CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
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

def select_action(policy, state, env, device, token_hidden_prob=0.3):
    """Given a policy network and the current environment state, select an action.
    The state is encoded using encode_board, then the network's logits are masked
    to zero out illegal moves. An action is then sampled from the resulting distribution."""

    valid_moves = env.get_valid_moves()
    if not valid_moves:
        return -1, torch.tensor(0.0, device=device)
    
    state_encoded = encode_board(state, token_hidden_prob)
    state_tensor = torch.tensor(state_encoded, dtype=torch.float32).unsqueeze(0).to(device)
    logits = policy(state_tensor).squeeze(0)  # shape: (7,)
    
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

def train_rl(policy, optimizer, num_games=100, token_hidden_prob=0.3, device=torch.device("cpu")):
   # Train the policy network by rewarding if there is a win and taking away points if there is a loss
    policy.train()
    for game in range(num_games):
        env = Connect4Env()
        log_probs = []  
        state = env.board.copy()
        
        while not env.game_over:
            if env.current_player == 1:
                # Check for legal moves
                if not env.get_valid_moves():
                    break
                action, log_prob = select_action(policy, state, env, device, token_hidden_prob)
                
                if action == -1:
                    break
                log_probs.append(log_prob)
            else:
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                action = random.choice(valid_moves)
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

def evaluate_rl(policy, num_games=20, token_hidden_prob=0.3, device=torch.device("cpu")):
    # Evaluate the nn
    policy.eval()
    wins, losses, draws = 0, 0, 0
    for game in range(num_games):
        env = Connect4Env()
        state = env.board.copy()
        while not env.game_over:
            if env.current_player == 1:
                # Check valid moves before selecting action.
                if not env.get_valid_moves():
                    break
                action, _ = select_action(policy, state, env, device, token_hidden_prob)
                if action == -1:
                    break
            else:
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                action = random.choice(valid_moves)
            env.make_move(action)
            state = env.board.copy()
            
        if env.winner == 1:
            wins += 1
        elif env.winner == 2:
            losses += 1
        else:
            draws += 1
    total = wins + losses + draws
    print(f"Evaluation over {total} Game: Wins: {wins}, Losses: {losses}, Draws: {draws}, Win Rate: {wins/total*100:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Connect4CNN().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    num_training_games = 1000
    print("Starting reinforcement learning training...")
    train_rl(policy, optimizer, num_games=num_training_games, token_hidden_prob=0.3, device=device)
    
    print("Evaluating trained RL agent...")
    evaluate_rl(policy, num_games=20, token_hidden_prob=0.3, device=device)

if __name__ == "__main__":
    main()
