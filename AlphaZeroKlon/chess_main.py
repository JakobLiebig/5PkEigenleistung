import numpy as np

import game_state
import gym
import gym_chess
from copy import deepcopy

class GameState(game_state.Base):
  def initial():
    state = gym.make('ChessAlphaZero-v0')
    board_state = state.reset()  

    return GameState(state, 0, board_state, False)
  
  def render():
    raise NotImplementedError

  def __init__(self, state, reward, board_state, done):
    self.state = state
    self.reward = reward
    self.board_state = board_state
    self.done = done

  def step(self, action):
    new_state = deepcopy(self.state)

    board_state, reward, done, _ = new_state.step(action)
    return GameState(new_state, reward, board_state, done)
  
  def is_terminal(self) -> bool:
    return self.done
  
  def get_reward(self) -> float:
    return self.reward
  
  def generate_state(self) -> np.array:
    return self.board_state

  def generate_mask(self) -> np.array:
    mask = np.zeros(4672, dtype=bool)
    mask[self.state.legal_actions] = True

    return mask
  
  def generate_possible_actions(self) -> np.array:
    return self.state.legal_actions

import torch
import torch.nn as nn

import neural_network

def masked_softmax(x, mask, sum_dim):
    exp_x = x.exp()
    zeros_like_x = torch.zeros_like(x)

    masked_exp_x = torch.where(mask, exp_x, zeros_like_x)
    masked_exp_x_sum = masked_exp_x.sum(sum_dim, keepdim=True)

    masked_softmax = torch.where(mask, exp_x / (masked_exp_x_sum + 1e-8), zeros_like_x)
    return masked_softmax

class NeuralNetwork(neural_network.Base):
  def __init__(self, state_shape, action_space, device):
      super().__init__()
      
      self.common = nn.Sequential(
          nn.Conv2d(state_shape[0], 64, (5), padding='same'),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 16, (3), padding='same'),
          nn.BatchNorm2d(16),
          nn.ReLU(),
          nn.Conv2d(16, 8, (3), padding='same'),
          nn.BatchNorm2d(8),
          nn.ReLU(),
          nn.Flatten()
      ).to(device)

      self.policy_stream = nn.Sequential(
          nn.Linear(7616, 7616),
          nn.ReLU(),
          nn.Linear(7616, action_space)
      ).to(device)
      
      self.value_stream = nn.Sequential(
          nn.Linear(7616, 7616),
          nn.ReLU(),
          nn.Linear(7616, 1)
      ).to(device)

      self.softmax = nn.Softmax(0)
  
      self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
      self.mse = nn.MSELoss()
      
      self.device = device
  
  def forward(self, x, mask):
      x = self.common(x)
      
      value = self.value_stream(x)
      policy = masked_softmax(self.policy_stream(x), mask, 1)
      policy = policy.masked_select(mask)
      
      return value, policy

  def __call__(self, x, mask):
      x = torch.from_numpy(x).to(self.device).float()
      mask = torch.from_numpy(mask).to(self.device)
      
      value, policy = self.forward(x, mask)
      
      value = value.detach().cpu().numpy()
      policy = policy.detach().cpu().numpy()
      
      return value, policy      
      
      
  def _calc_loss(self, *examples):
      x, mask, value_target, policy_target = [torch.from_numpy(x).to(self.device) for x in examples]
      
      value, policy = self.forward(x.float(), mask)
      
      loss = self.mse(value, value_target.float()) + self.mse(policy, policy_target.float())
      return loss
      
  def fit(self, examples):
      x, mask, value_target, policy_target = examples
      
      x = np.stack(x)
      mask = np.stack(mask)
      value_target = np.stack(value_target)
      policy_target = np.concatenate(policy_target, 0)
      
      loss = self._calc_loss(x, mask, value_target, policy_target)
      
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      
      loss = float(loss.detach().cpu().numpy())
      return loss
    
  def save_to(self, file_path):
      state_dict = self.state_dict()
      torch.save(state_dict, file_path)
  
  def load_from(self, file_path, map_location):
      state_dict = torch.load(file_path, map_location)
      self.load_state_dict(state_dict)
    
import coach
import logger
import matplotlib.pyplot as plt
from copy import deepcopy

initial_state = GameState.initial()

buffer_size = 1000
batch_size = 4

mcts_iterations = 200
train_temperature = 2.
eval_temperature = 4.

iterations = 1000000
train_iterations = 75
eval_iterations = 25

current_best = NeuralNetwork([8, 8, 119], 4672, torch.device('cuda'))

save_path = "/content/trained/model.bin"

class Logger(logger.Base):
    def __init__(self):
      self.losses = []

    def log(self, t, m):
        if 'event start' in t:
          print("[+] ", t, m.split(" ")[0])
        elif 'event stop' in t:
          print("[-] ", t, m.split(" ")[0])
          
          if 'training' in t:
            plt.plot(self.losses)
            plt.ylabel('loss')
            plt.show()
        elif 'loss' in t:
          print("[!] current loss: ", m)
          self.losses.append(float(m))

logger_ = Logger()
trainer = coach.Coach(initial_state, buffer_size, logger_)

for index in range(iterations):
    new_contestant = deepcopy(current_best)
    trainer.train(new_contestant, train_iterations, batch_size, train_temperature, mcts_iterations)
    
    current_best, win_perc = trainer.pit([current_best, new_contestant], eval_iterations, eval_temperature, mcts_iterations)
    current_best.save_to(save_path)

    print(f"[!] {win_perc}")