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

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight)
        
        if self.args.use_kaiming_init:
            self.linear.apply(init_weights)
            self.sketch_linear.apply(init_weights)
            
    def extract_feature(self, batch, num):
        sketch_img = batch[f'sketch_img_{num}'].to(device)
        positive_img = batch[f'positive_img_{num}'].to(device)
        
        positive_feature, fm_6b_pos = self.sample_embedding_network(positive_img)
        sketch_feature, fm_6b_ske = self.sketch_embedding_network(sketch_img)
        
        positive_feature = self.attention(positive_feature)
        sketch_feature = self.sketch_attention(sketch_feature)
        
        positive_feature = self.linear(positive_feature)
        sketch_feature = self.sketch_linear(sketch_feature)
        
        fm_6bs = {
            "fm_6b_pos": fm_6b_pos,
            "fm_6b_ske": fm_6b_ske
        }
        
        return sketch_feature, positive_feature, fm_6bs
    
    def forward(self, batch):
        sketch_feature_1, positive_feature_1, fm_6bs_1 = self.extract_feature(batch=batch, num=1)
        sketch_feature_2, positive_feature_2, fm_6bs_2 = self.extract_feature(batch=batch, num=2)
        
        negative_img = batch['negative_img'].to(device)
        negative_feature, _ = self.sample_embedding_network(negative_img)
        negative_feature = self.linear(self.attention(negative_feature))
        
        return {
            'sketch_feature_1': sketch_feature_1, 'sketch_feature_2': sketch_feature_2,
            'positive_feature_1': positive_feature_1, 'positive_feature_2': positive_feature_2,
            'negative_feature': negative_feature,
            'fm_6bs_1': fm_6bs_1, 'fm_6bs_2': fm_6bs_2
        }
    