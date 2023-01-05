# imports 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import os 

class QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x 
    
    def save(self, file_name = 'model.pth'):
        folder_path = './model'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:

    def __init__(self, model, lr, gamma):
        self.lr = lr 
        self.gamma = gamma 
        self.model = model 
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss() 
    
    def train_step(self, state, input, reward, next_state, game_result):
        state = torch.tensor(state, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        input = torch.tensor(input, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float) 

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            input = torch.unsqueeze(input, 0)
            reward = torch.unsqueeze(reward, 0)
            game_result = (game_result,)
        
        # predicted Q values for current state 
        pred = self.model(state)

        # bellman equation 
        target = pred.clone()
        for idx in range(len(game_result)):
            q_new = reward[idx]
            if game_result[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(input).item()] = q_new 

        # empty gradients 
        self.optimizer.zero_grad()

        # calculate loss 
        loss = self.criterion(target, pred)
        loss.backward() 

        self.optimizer.step()