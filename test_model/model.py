import torch
import torch.nn as nn

from baseline.backbones import InceptionV3
from baseline.attention import Linear_global, SelfAttention, SketchAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.attn = SketchAttention(args)
        self.proj = Linear_global(feature_num=64)
        
    def forward(self, x):
        x = self.bilstm(x)
        x = self.attn(x)
        x = self.proj(x)
        
        return x

class Siamese_SBIR(nn.Module):
    def __init__(self, args):
        super(Siamese_SBIR, self).__init__()
        self.args = args
        self.sample_embedding_network = InceptionV3(args=args)
        self.attention = SelfAttention(args)
        self.linear = Linear_global(feature_num=64)

        self.sketch_embedding_network = InceptionV3(args=args)
        self.sketch_attention = SelfAttention(args)
        
        self.sample_embedding_network.fix_weights()
        self.sketch_embedding_network.fix_weights()
        self.attention.fix_weights()
        self.sketch_attention.fix_weights()
        self.linear.fix_weights()
        
        self.bilstm = BiLSTM(args)        

    def forward(self, batch):
        positive_img = batch['positive_img'].to(device)
        negative_img = batch['negative_img'].to(device)
        sketch_imgs = batch['sketch_imgs'].to(device)

        positive_feature, fm_6b_pos = self.sample_embedding_network(positive_img) # (1, 2048)
        negative_feature, _ = self.sample_embedding_network(negative_img) # (1, 2048)
        sketch_features, fm_6b_ske = self.sketch_embedding_network(sketch_imgs) # (20, 2048)

        positive_feature = self.linear(self.attention(positive_feature))
        negative_feature = self.linear(self.attention(negative_feature))
        sketch_features = self.sketch_linear(self.sketch_attention(sketch_features))
        
        fm_6bs = {
            "fm_6b_pos": fm_6b_pos,
            "fm_6b_ske": fm_6b_ske
        }
        
        return {
            "positive_feature": positive_feature, 
            "negative_feature": negative_feature,
            "sketch_features": sketch_features,
            "fm_6bs": fm_6bs
        }
