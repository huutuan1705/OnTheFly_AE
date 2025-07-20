import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.backbones import InceptionV3
from baseline.attention import Linear_global, SelfAttention
from baseline.transformer import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Siamese_SBIR(nn.Module):
    def __init__(self, args):
        super(Siamese_SBIR, self).__init__()
        self.args = args
        self.sample_embedding_network = InceptionV3(args=args)
        self.attention = Encoder()
        
        self.sketch_embedding_network = InceptionV3(args=args)
        self.sketch_attention = Encoder()
    
            
    def forward(self, batch):
        sketch_img = batch['sketch_img']
        positive_img = batch['positive_img']
        negative_img = batch['negative_img']
        
        positive_feature = self.sample_embedding_network(positive_img).to(device)
        negative_feature = self.sample_embedding_network(negative_img).to(device)
        sketch_feature = self.sketch_embedding_network(sketch_img).to(device)
        
        positive_feature, _ = self.attention(positive_feature).to(device)
        negative_feature, _ = self.attention(negative_feature).to(device)
        sketch_feature, _ = self.sketch_attention(sketch_feature).to(device)
        
        return sketch_feature, positive_feature, negative_feature
    