import flappy_bird_gymnasium
import gymnasium as gym
import dqn as DQN
from experience_replay import ReplayMemory
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

if torch.backends.nps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda" 
else:
    device = "cpu"
    
class Agent:
    def __ini__(self, param_set):
        self.param_set = param_set
        
        with open("param_set.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]
            
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_thresold = params["reward_thresold"]
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
    
    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=True)
        
        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n
        
        policy_dqn = DQN(num_state, num_action).to(device)
        
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = DQN(num_state, num_action).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            steps = 0
            
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            
            
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            episode_reward = 0
            terminated = False
            
            while not terminated:
                 # Action
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()  #explore
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() #exploit
                
                # Processing:
                next_state, reward, terminated, _, _ = env.step(action.item()) 
                
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
            
                if is_training:
                    memory.append(state, action, new_state, reward, terminated)
                    steps += 1
                    
                state = next_state
                episode_reward += reward   
            
            print(f"episode = {episode+1}  with total reward = {episode_reward} & epsilon = {epsilon}")
            
            if is_training:
                # Epsilon Decay
                epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
                
            if is_training and len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                
                optimize(mini_batch, policy_dqn, target_dqn)
                
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step = 0