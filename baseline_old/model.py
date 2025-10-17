import torch
import torch.nn as nn

from baseline.backbones import InceptionV3
from baseline.attention import Linear_global, Attention_global

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.embedding_network = InceptionV3(args=args)
        self.attention = Attention_global(args)
        self.linear = Linear_global(feature_num=self.args.output_size)
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight)
        
        if self.args.use_kaiming_init:
            self.attention.apply(init_weights)
            self.linear.apply(init_weights)
            
    def forward(self, batch):
        sketch_img = batch['sketch_img'].to(device)
        positive_img = batch['positive_img'].to(device)
        negative_img = batch['negative_img'].to(device)
        
        positive_feature = self.linear(self.attention(self.embedding_network(positive_img)))
        negative_feature = self.linear(self.attention(self.embedding_network(negative_img)))
        sketch_feature = self.linear(self.attention(self.embedding_network(sketch_img)))
        
        return sketch_feature, positive_feature, negative_feature