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
        # self.sketch_embedding_network.fix_weights()
        self.attention.fix_weights()
        # self.sketch_attention.fix_weights()
        self.linear.fix_weights()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight)

        if self.args.use_kaiming_init:
            self.sketch_linear.apply(init_weights)

    def forward(self, batch):
        positive_img = batch['positive_img'].to(device)
        negative_img = batch['negative_img'].to(device)
        sketch_imgs_1 = batch['sketch_imgs_1'].squeeze(0).to(device)
        sketch_imgs_2 = batch['sketch_imgs_2'].squeeze(0).to(device)

        positive_feature, _ = self.sample_embedding_network(positive_img) # (1, 2048)
        negative_feature, _ = self.sample_embedding_network(negative_img) # (1, 2048)
        sketch_features_1, _ = self.sketch_embedding_network(sketch_imgs_1) # (20, 2048)
        sketch_features_2, _ = self.sketch_embedding_network(sketch_imgs_2) # (20, 2048)

        positive_feature = self.linear(self.attention(positive_feature))
        negative_feature = self.linear(self.attention(negative_feature))
        sketch_features_1 = self.sketch_linear(self.sketch_attention(sketch_features_1))
        sketch_features_2 = self.sketch_linear(self.sketch_attention(sketch_features_2))
        
        return {
            "positive_feature": positive_feature, "negative_feature": negative_feature,
            "sketch_features_1": sketch_features_1, "sketch_features_2": sketch_features_2
        }
