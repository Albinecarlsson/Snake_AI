import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# set torch default to GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear3 = nn.Linear(hidden_size, 256)
        self.linear2 = nn.Linear(256, output_size)
        
    #Predict function       
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    #Save function which saves the model
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)    
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
    #Load function which loads the model from given path
    def load(self, file_name='model.pth'):
        #model_folder_path = './model'
        #file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()


class Trainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
       
    #Trains the model with the given data    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # if we only have one state We unsqueeze to get batch size of one
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        #predict Q values with current state
        pred = self.model(state)
        target = pred.clone()
        
        # Calculate the Q_new value 
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        
        # compare and update Q values 
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()