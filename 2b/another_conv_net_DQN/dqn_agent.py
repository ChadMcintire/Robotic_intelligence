import torch
from torch import nn
import copy
from collections import deque
import random

class DQN_Agent:

    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size):
        #set the random seeds
        torch.manual_seed(seed)

        #build neural net, and target network for stability and 
        #to avoid catastrophic forgetting
        self.q_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.q_net)
        
        #set the device to cpu or cuda if it is available 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net.to(self.device)
        self.target_net.to(self.device)

        #loss and optimizer
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        #how often to sync with target network
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        
        #set up discount
        self.gamma = torch.tensor(0.95).float().to(self.device)

        #set up replay
        self.experience_replay = deque(maxlen=exp_replay_size)
        return

    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)

    #load for inference
    def load_pretrained_model(self, model_path):
        self.q_net.load_state_dict(torch.load(model_path))

    #save for inference
    def save_trained_model(self, model_path="cartpole-dqn.pth"):
        torch.save(self.q_net.state_dict(), model_path)

    
    def get_action(self, state, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float().to(self.device))
        Q, A = torch.max(Qp, axis=0)
        # episilon greedy action, action is selected from NN if random number greater than episilon 
        A = A if torch.rand(1, ).item() > epsilon else torch.randint(0, action_space_len, (1,))
        return A

    #get targets 
    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q

    #collect experience tuple
    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return

    #sample from experiences, return state, action, next_reward, next_state
    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn

    def train(self, batch_size):
        s, a, rn, sn = self.sample_from_experience(sample_size=batch_size)
        #up target network if sync counter = the sync freqency
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0

        # predict expected return of current state using main network
        qp = self.q_net(s.to(self.device))
        pred_return, _ = torch.max(qp, axis=1)

        # get target return using target network
        q_next = self.get_q_next(sn.to(self.device))

        #update using the Bellman equation
        target_return = rn.to(self.device) + self.gamma * q_next
        
        #calculate loss and back prop
        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()
