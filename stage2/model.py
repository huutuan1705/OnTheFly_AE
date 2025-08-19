import torch
import torch.nn as nn

from baseline.backbones import InceptionV3
from baseline.attention import Linear_global, SelfAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Siamese_SBIR(nn.Module):
    def __init__(self, args):
        super(Siamese_SBIR, self).__init__()
        self.args = args
        self.sample_embedding_network = InceptionV3(args=args)
        self.attention = SelfAttention(args)
        self.linear = Linear_global(feature_num=64)
        
        self.sketch_embedding_network = InceptionV3(args=args)
        self.sketch_attention = SelfAttention(args)
        self.sketch_linear = Linear_global(feature_num=64)
        
        self.sample_embedding_network.fix_weights()
        self.sketch_embedding_network.fix_weights()
        self.attention.fix_weights()
        self.sketch_attention.fix_weights()
        self.linear.fix_weights()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight)
        
        if self.args.use_kaiming_init:
            self.sketch_linear.apply(init_weights)
            
    def forward(self, batch):
        positive_img = batch['positive_img'].to(device)
        negative_img = batch['negative_img'].to(device)
        sketch_imgs_1 = batch['sketch_imgs_1'].to(device)
        sketch_imgs_2 = batch['sketch_imgs_2'].to(device)
        
        positive_feature, _ = self.sample_embedding_network(positive_img)
        negative_feature, _ = self.sample_embedding_network(negative_img)
        sketch_feature_1, _ = self.sketch_embedding_network(sketch_imgs_1)
        sketch_feature_2, _ = self.sketch_embedding_network(sketch_imgs_2)
        
        return