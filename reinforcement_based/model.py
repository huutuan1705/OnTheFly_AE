import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.backbones import InceptionV3
from baseline.attention import Linear_global, SelfAttention

class Policy(nn.Module):
    def __init__(self, state_dim = 2048, action_dim = 64, log_std=0):
        super(Policy, self).__init__()
        self.actor = nn.Linear(2048, 64)
        self.action_log_std = nn.Parameter(torch.ones(action_dim) * log_std)

    def forward(self, x):
        action_mean = self.actor(x)
        return action_mean

    def fix_network(self):
        for name, x in self.named_parameters():
            if name in ['action_log_std']:
                x.requires_grad = False
                print(name, x.requires_grad)

    def select_action(self, x):
        action_mean = self.forward(x)
        m = torch.distributions.Normal(action_mean, torch.exp(0.5*self.action_log_std))
        sketch_anchor_embedding = m.sample()
        log_prob = m.log_prob(sketch_anchor_embedding).sum()
        entropy = m.entropy()
        return action_mean, sketch_anchor_embedding, log_prob, entropy
    
class RL_based(nn.Module):
    def __init__(self, args):
        super(RL_based, self).__init__()
        self.args = args
        self.sample_embedding_network = InceptionV3(args=args)
        self.attention = SelfAttention(args)
        self.linear = Linear_global(feature_num=args.output_size)

        self.sketch_embedding_network = InceptionV3(args=args)
        self.sketch_attention = SelfAttention(args)
        self.sketch_linear = Linear_global(feature_num=args.output_size)
        
        self.sample_embedding_network.fix_weights()
        self.sketch_embedding_network.fix_weights()
        self.attention.fix_weights()
        self.sketch_attention.fix_weights()
        self.linear.fix_weights()
        
        self.policy_network = Policy()
    
    def get_reward(self, action, sketch_name):
        sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
        position_query = self.Image_Name_Train.index(sketch_query_name)
        target_distance = F.pairwise_distance(F.normalize(action),
                                              self.Image_Array_Train[position_query])
        distance = F.pairwise_distance(F.normalize(action), self.Image_Array_Train)
        rank = distance.le(target_distance).sum()

        if rank.item() == 0:
            reward = 1.
        else:
            reward = 1. / rank.item()
        return reward
       
    def calculate_loss(self, log_probs, rewards, entropies):
        loss = 0
        gamma = 0.9
        for i in reversed(range(len(rewards))):
            R = gamma ** (len(rewards) - i -1) * rewards[i]
            # R =  rewards[i] # Flat Reward
            loss = loss - log_probs[i] * R - 0.0001 * entropies[i]
        loss = loss / len(rewards)
        return loss