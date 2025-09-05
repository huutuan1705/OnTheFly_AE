import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.pool_method =  nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(2048)
        self.mha = nn.MultiheadAttention(2048, num_heads=args.num_heads, batch_first=True)
        # self.mha = nn.MultiheadAttention(2048, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        identify = x
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h*w).transpose(1, 2)
        x_att = self.norm(x_att)
        
        att_out, _  = self.mha(x_att, x_att, x_att)
        att_out = self.dropout(att_out)
        att_out = att_out.transpose(1, 2).reshape(bs, c, h, w)
        
        output = identify * att_out + identify
        output = self.pool_method(output).view(-1, 2048)
        
        return F.normalize(output)
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(2048, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class SketchAttention(nn.Module):
    def __init__(self, args):
        super(SketchAttention, self).__init__()
        self.norm = nn.LayerNorm(2048)
        self.mha = nn.MultiheadAttention(2048, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        identify = x
        x_att = self.norm(x)
        att_out, _  = self.mha(x_att, x_att, x_att)
        att_out = self.dropout(att_out)
        
        output = identify * att_out + identify
        output = F.normalize(output)
        return output
        
        
class Linear_global(nn.Module):
    def __init__(self, feature_num):
        super(Linear_global, self).__init__()
        self.head_layer = nn.Linear(2048, feature_num)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(x)
        return F.normalize(self.head_layer(x))
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False

if __name__ == "__main__":
    dim = 2048
    dt_rank = 4
    dim_inner = 32
    d_state = 8
    
    model = SketchAttention(None)
    x = torch.randn(2, 5, dim)
    out = model(x)
    print("Output shape:", out.shape)