import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# from zeta import SSM

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

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd.floor_()
        return x.div(keep) * rnd

class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, p=0.0):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def build_rel_pos_index(H, W):
    # (L, L) index map into a (2H-1)*(2W-1) table
    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=0)  # (2, H, W)
    coords = coords.reshape(2, -1)  # (2, L)
    rel = coords[:, :, None] - coords[:, None, :]  # (2, L, L)
    rel[0] += H - 1
    rel[1] += W - 1
    rel_index = rel[0] * (2 * W - 1) + rel[1]  # (L, L) in [0, (2H-1)*(2W-1)-1]
    return rel_index  # (L, L)

class SelfAttention2D(nn.Module):
    def __init__(self, args, num_heads=8, in_ch=2048, H=8, W=8, d_model=512,
                 attn_drop=0.0, proj_drop=0.2, drop_path=0.0):
        super(SelfAttention2D, self).__init__()
        self.in_ch, self.H, self.W = in_ch, H, W
        self.L = H * W
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.in_proj = nn.Linear(in_ch, d_model, bias=True)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        # 🔧 giữ ở d_model để cộng residual với tokens (d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        table_size = (2 * H - 1) * (2 * W - 1)
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, table_size))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        self.register_buffer("rel_index", build_rel_pos_index(H, W), persistent=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, hidden_mult=4, p=proj_drop)

        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_mlp  = nn.Parameter(torch.zeros(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Attention pooling và chiếu về 2048 ở đây
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, in_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_ch and H == self.H and W == self.W

        tokens = x.permute(0, 2, 3, 1).reshape(B, self.L, C)        # (B, L, 2048)
        tokens = self.in_proj(tokens)                                # (B, L, d_model)

        y = self.norm1(tokens)
        qkv = self.qkv(y).reshape(B, self.L, 3, self.num_heads, self.d_head).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]                             # (B, h, L, dh)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)    # (B, h, L, L)
        bias = self.rel_pos_bias[:, self.rel_index.view(-1)].view(self.num_heads, self.L, self.L)
        attn = (attn + bias.unsqueeze(0)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        z = attn @ v                                                # (B, h, L, dh)
        z = z.transpose(1, 2).reshape(B, self.L, self.d_model)      # (B, L, d_model)
        z = self.out_proj(z)                                        # (B, L, d_model)
        z = self.proj_drop(z)
        tokens = tokens + self.drop_path(self.gamma_attn * z)       # ✅ cùng chiều (B,L,d_model)

        y2 = self.norm2(tokens)
        z2 = self.mlp(y2)                                           # (B, L, d_model)
        tokens = tokens + self.drop_path(self.gamma_mlp * z2)

        # Attention pooling -> (B, d_model) -> chiếu về 2048
        q_pool = self.pool_query.expand(B, -1, -1)                  # (B,1,d_model)
        k_pool = self.pool_norm(tokens)
        v_pool = tokens
        attn_pool = (q_pool @ k_pool.transpose(1, 2)) / math.sqrt(self.d_model)  # (B,1,L)
        attn_pool = attn_pool.softmax(dim=-1)
        pooled = (attn_pool @ v_pool).squeeze(1)                    # (B, d_model)
        pooled = self.pool_proj(pooled)                             # (B, 2048)

        return F.normalize(pooled, dim=1)

    
# class SSMAttention(nn.Module):
#     def __init__(self, args):
#         super(SSMAttention, self).__init__() 
#         self.pool_method =  nn.AdaptiveAvgPool2d(1)
#         self.norm = nn.LayerNorm(2048)  
#         self.ssm = SSM(in_features=2048, dt_rank=16, dim_inner=2048, d_state=8) 
#         self.dropout = nn.Dropout(p=0.2)
        
#     def forward(self, x):
#         identify = x
#         bs, c, h, w = x.shape
#         x_att = x.reshape(bs, c, h*w).transpose(1, 2)
#         x_att = self.norm(x_att)
        
#         att_out = self.ssm(x_att)
#         att_out = self.dropout(att_out)
#         att_out = att_out.transpose(1, 2).reshape(bs, c, h, w)
        
#         output = identify * att_out + identify
#         output = self.pool_method(output).view(-1, 2048)
#         return F.normalize(output)
    
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
    dim = 16
    dt_rank = 4
    dim_inner = 32
    d_state = 8
    
    # model = SSMAttention(None, dim, dt_rank, dim_inner, d_state)
    # x = torch.randn(2, 5, dim)
    # out = model(x)
    # print("Output shape:", out.shape)