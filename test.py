import torch
import torch.nn as nn
from zeta import SSM

class SSMAttention(nn.Module):
    def __init__(self, args, dim, dt_rank, dim_inner, d_state):
        super(SSMAttention, self).__init__()   
        self.ssm = SSM(in_features=dim, dt_rank=dt_rank, dim_inner=dim_inner, d_state=d_state) 
        
    def forward(self, x):
        x = self.ssm(x)
        return x
    

if __name__ == "__main__":
    dim = 2048
    dt_rank = 4
    dim_inner = 2048
    d_state = 8
    
    model = SSMAttention(None, dim, dt_rank, dim_inner, d_state)
    x = torch.randn(24, 64, 2048)
    out = model(x)
    print("Output shape:", out.shape)