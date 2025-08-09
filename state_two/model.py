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
        
    def extract_feature(self, batch, num):
        sketch_imgs = batch[f'sketch_imgs_{num}'].to(device)
        positive_img = batch[f'positive_img_{num}'].to(device)
        negative_img = batch[f'negative_img_{num}'].to(device)
        
        positive_feature, _ = self.sample_embedding_network(positive_img)
        negative_feature, _ = self.sample_embedding_network(negative_img)
        
        positive_feature = self.attention(positive_feature)
        negative_feature = self.attention(negative_feature)
        
        positive_feature = self.linear(positive_feature)
        negative_feature = self.linear(negative_feature)
        
        sketch_features = []
        
        for sketch_img in sketch_imgs:
            sketch_feature, _ = self.sketch_embedding_network(sketch_img)
            sketch_feature = self.sketch_attention(sketch_feature)
            sketch_feature = self.sketch_linear(sketch_feature)
            sketch_features.append(sketch_feature)
        return sketch_features, positive_feature, negative_feature
    
    
    def forward(self, batch):
        sketch_feature_1, positive_feature_1, negative_feature_1 = self.extract_feature(batch=batch, num=1)
        sketch_feature_2, positive_feature_2, negative_feature_2 = self.extract_feature(batch=batch, num=2)
        
        return {
            'sketch_features_1': sketch_feature_1, 'sketch_features_2': sketch_feature_2,
            'positive_feature_1': positive_feature_1, 'positive_feature_2': positive_feature_2,
            'negative_feature_1': negative_feature_1, 'negative_feature_2': negative_feature_2,
        }