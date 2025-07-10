import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Sử dụng MultiHeadAttention có sẵn của PyTorch

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention với causal mask
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, key_padding_mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim=2048, tgt_vocab_size=None, d_model=64, n_heads=8, 
                 n_layers=1, d_ff=2048, max_seq_length=5000, dropout=0.1, 
                 use_embedding=False):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.use_embedding = use_embedding
        
        # Projection layer cho feature maps từ CNN
        if not use_embedding:
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            # Embedding layers (cho text input)
            self.encoder_embedding = nn.Embedding(input_dim, d_model)
            if tgt_vocab_size is not None:
                self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 2D positional encoding cho feature maps
        self.pos_encoding_2d = nn.Parameter(torch.randn(1, 64, d_model))  # 8x8 = 64 positions
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Final linear layer
        if tgt_vocab_size is not None:
            self.fc = nn.Linear(d_model, tgt_vocab_size)
        else:
            self.fc = nn.Linear(d_model, d_model)  # hoặc output dimension khác
        self.dropout = nn.Dropout(dropout)
        
    def prepare_feature_maps(self, feature_maps):
        """
        Chuyển đổi feature maps (B, C, H, W) thành sequence (B, seq_len, d_model)
        feature_maps: (batch_size, 2048, 8, 8)
        """
        batch_size, channels, height, width = feature_maps.shape
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        feature_maps = feature_maps.view(batch_size, channels, -1).transpose(1, 2)
        
        # Project to model dimension: (B, H*W, C) -> (B, H*W, d_model)
        projected = self.input_projection(feature_maps)
        
        # Add 2D positional encoding
        seq_len = projected.size(1)
        pos_encoding = self.pos_encoding_2d[:, :seq_len, :]
        projected = projected + pos_encoding
        
        return self.dropout(projected)
        
    def forward(self, src, tgt=None, src_pad_mask=None, tgt_pad_mask=None):
        if self.use_embedding:
            # Trường hợp input là token IDs
            src_emb = self.dropout(self.positional_encoding(
                self.encoder_embedding(src) * math.sqrt(self.d_model)))
            if tgt is not None:
                tgt_emb = self.dropout(self.positional_encoding(
                    self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        else:
            # Trường hợp input là feature maps từ CNN
            src_emb = self.prepare_feature_maps(src)
            if tgt is not None:
                if hasattr(self, 'decoder_embedding'):
                    tgt_emb = self.dropout(self.positional_encoding(
                        self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
                else:
                    tgt_emb = self.prepare_feature_maps(tgt)
        
        # Encoder
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_pad_mask)
        
        # Nếu chỉ có encoder (encoder-only model)
        if tgt is None:
            output = self.fc(enc_output)
            return F.normalize(output)
        
        # Decoder với causal mask
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(1))
        if tgt_emb.is_cuda:
            tgt_mask = tgt_mask.cuda()
            
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_pad_mask, tgt_mask)
        
        # Output projection
        output = self.fc(dec_output)
        
        return F.normalize(output)

    def generate_square_subsequent_mask(self, sz):
        """Tạo causal mask cho decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def create_pad_mask(self, seq, pad_idx=0):
        """Tạo padding mask"""
        return (seq == pad_idx)
    
if __name__ == "__main__":
    input_dim = 2048  # Channels của feature maps
    d_model = 512
    n_heads = 8
    n_layers = 1
    d_ff = 2048
    dropout = 0.1
    
    # Tạo model (encoder-only)
    model = Transformer(
        input_dim=input_dim,
        tgt_vocab_size=None,  # Không cần target vocab cho encoder-only
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        use_embedding=False  # Không sử dụng embedding
    )
    
    # Tạo dữ liệu test - feature maps từ CNN
    batch_size = 16
    feature_maps = torch.randn(batch_size, 2048, 8, 8)  # (B, C, H, W)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(feature_maps)  # Chỉ encoder
        print(f"Input shape: {feature_maps.shape}")
        print(f"Output shape: {output.shape}")  # (B, 64, d_model)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    