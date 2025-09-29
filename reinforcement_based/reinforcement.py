import torch
import torch.nn as nn

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