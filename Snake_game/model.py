import torch 
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# set torch default to GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 512)
        self.linear4 = nn.Linear(512, output_size)     
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear4(x)
        return x
    
    # saving the best weigths. 
    def save(self, file_name='test.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        print("saved!")
    
    def load(self, file_name='test.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()
        for param in self.parameters():
            print(param)
        print("loaded!")
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()      
    def train_step(self, state, action, reward, next_state, done):
        # make all input params to tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        # if we only have one state we convert the state to allowed state by unsqueezeing
        print(state)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
                
        #---  Bellman equation  ---#  
        #Get predicted Q values for current state
        print(state)
        pred = self.model(state)
        target = pred.clone()
        
        # Calculate Q_new values for each of the batch values
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new

        # in pytorch you allways have to set zero_grad() between interations
        # Run the MSE function and backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
        