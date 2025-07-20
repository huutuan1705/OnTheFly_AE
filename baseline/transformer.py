import math
import os
import copy

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, LayerNorm, Softmax, GELU

PATCH_SIZE = 1
HIDDEN_SIZE = 64
MLP_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 1
ATTENTION_DROPOUT_RATE = 0.0
DROPOUT_RATE = 0.1

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__() 
        self.num_attention_heads = NUM_HEADS
        self.attention_head_size = int(HIDDEN_SIZE / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = Linear(HIDDEN_SIZE, self.all_head_size)
        self.key = Linear(HIDDEN_SIZE, self.all_head_size)
        self.value = Linear(HIDDEN_SIZE, self.all_head_size)
        
        self.out = Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.attention_dropout = Dropout(ATTENTION_DROPOUT_RATE)
        self.proj_dropout = Dropout(DROPOUT_RATE)
        
        self.softmax = Softmax(dim=-1)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        
        # This is actually dropping out entire tokens to attend to, which might
        weights = attention_probs
        attention_probs = self.attention_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
    

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = Linear(HIDDEN_SIZE, MLP_DIM)
        self.fc2 = Linear(MLP_DIM, HIDDEN_SIZE)
        self.act = GELU()
        self.dropout = Dropout(DROPOUT_RATE)
        
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
        
class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()
        patch_size = PATCH_SIZE
        n_patches = (8 // patch_size) * (8 // patch_size)
        self.pos_embeddings = nn.parameter.Parameter(torch.zeros(1, n_patches + 1, HIDDEN_SIZE))
        self.dropout = Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        
        print(x.shape)
        print(self.pos_embeddings.shape)
        embeddings = x + self.pos_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.mlp_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.mlp = Mlp()
        self.attention = Attention()
        
    def forward(self, x):
        residual = x
        x = self.attention_norm(x)
        x, weights = self.attention(x)
        x = x + residual
        
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + residual
        return x, weights
    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(HIDDEN_SIZE, eps=1e-6)
        for _ in range(NUM_LAYERS):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))
    
    def forward(self, hidden_states):
        attention_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attention_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attention_weights
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings()
        self.encoder = Encoder()
    
    def forward(self, x):
        embedding_output = self.embeddings(x)
        return self.encoder(embedding_output)
    