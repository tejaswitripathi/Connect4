import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from Connect4Env import Connect4Env  
from connect4_bot import Connect4Bot   
from connect4_agents import RandomAgent

def encode_board(board, token_hidden_prob=0.4):
    """
    Encode a Connect 4 board into a 3-channel tensor:
      - Channel 1: Player 1 tokens (visible if not hidden)
      - Channel 2: Player 2 tokens (visible if not hidden)
      - Channel 3: Hidden mask (1 if the cell is hidden, 0 otherwise)
    """
    channels = np.zeros((3, board.shape[0], board.shape[1]), dtype=np.float32)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            cell = board[i, j]
            if cell == 1 or cell == 2:
                if random.random() < token_hidden_prob:
                    channels[2, i, j] = 1.0  
                else:
                    if cell == 1:
                        channels[0, i, j] = 1.0
                    elif cell == 2:
                        channels[1, i, j] = 1.0
    return channels

def generate_data(num_games=10, token_hidden_prob=0.3):
    """
    Simulate Connect 4 games between two agents:
      - Player 1: minimax agent (agent type 0)
      - Player 2: random agent (agent type 1)
    
    For each move, record:
      - The encoded board state (3 channels)
      - The move chosen by the agent
      - Metadata indicating which agent made the move (0 for minimax, 1 for random)
    """
    data = []
    labels = []
    meta = []  
    for game in range(num_games):
        print(f"Generating data for game {game+1}/{num_games}...")
        env = Connect4Env()
        minimax_bot = Connect4Bot(max_depth=6)
        random_agent = RandomAgent()
        while not env.game_over:
            board_state = env.board.copy()
            encoded = encode_board(board_state, token_hidden_prob)
            if env.current_player == 1:
                move = minimax_bot.get_move(env, env.current_player)
                agent_type = 0  # minimax agent
            else:
                move = random_agent.get_move(env, env.current_player)
                agent_type = 1  # random agent
            if move == -1:
                break
            data.append(encoded)
            labels.append(move)
            meta.append(agent_type)
            env.make_move(move)
    return np.array(data), np.array(labels), np.array(meta)

# Define the CNN architecture for Connect 4.
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
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

def train(model, optimizer, criterion, train_loader, epochs=100, print_interval=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0  
        total_loss = 0.0    
        for i, (inputs, labels, _) in enumerate(train_loader, 1):  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_loss += loss.item()
            
            if i % print_interval == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(train_loader)}], '
                      f'Running Loss: {running_loss/print_interval:.4f}')
                running_loss = 0.0
        
        avg_loss = total_loss / len(train_loader)
        print(f'==> Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}')

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a given dataloader.
    Returns overall accuracy and breakdown by agent type.
    """
    model.eval()
    total_samples = 0
    correct = 0
    correct_minimax = 0
    total_minimax = 0
    correct_random = 0
    total_random = 0

    with torch.no_grad():
        for inputs, labels, meta in dataloader:
            inputs, labels, meta = inputs.to(device), labels.to(device), meta.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Evaluate accuracy
            for pred, target, agent in zip(predictions, labels, meta):
                if agent.item() == 0:
                    total_minimax += 1
                    if pred == target:
                        correct_minimax += 1
                else:  
                    total_random += 1
                    if pred == target:
                        correct_random += 1

    overall_acc = correct / total_samples if total_samples > 0 else 0
    minimax_acc = correct_minimax / total_minimax if total_minimax > 0 else 0
    random_acc = correct_random / total_random if total_random > 0 else 0
    return overall_acc, minimax_acc, random_acc

def main():
    print("Generating training data from minimax vs random self-play...")
    data, labels, meta = generate_data(num_games=10, token_hidden_prob=0.3)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}, Meta shape: {meta.shape}")
    
    inputs_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    meta_tensor = torch.tensor(meta, dtype=torch.long)
    
    dataset = TensorDataset(inputs_tensor, labels_tensor, meta_tensor)
    
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        running_loss = 0.0
        total_loss = 0.0
        for i, (inputs, targets, _) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/10], Batch [{i}/{len(train_loader)}], '
                      f'Running Loss: {running_loss/10:.4f}')
                running_loss = 0.0
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/10] Average Loss: {avg_loss:.4f}')
    
    train_overall, train_minimax, train_random = evaluate_model(model, train_loader, device)
    test_overall, test_minimax, test_random = evaluate_model(model, test_loader, device)
    
    print(f"Training Accuracy: {train_overall*100:.2f}% (Minimax: {train_minimax*100:.2f}%, Random: {train_random*100:.2f}%)")
    print(f"Testing Accuracy: {test_overall*100:.2f}% (Minimax: {test_minimax*100:.2f}%, Random: {test_random*100:.2f}%)")

if __name__ == "__main__":
    main()
