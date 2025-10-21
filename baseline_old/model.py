import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import optim
from tqdm import tqdm

from baseline_old.backbones import VGG16, ResNet50, InceptionV3
from baseline_old.attention import Linear_global, SelfAttention, Attention_global

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + "(args)")
        self.loss = nn.TripletMarginLoss(margin=args.margin)        
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.args = args
        
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        
        # self.attention = SelfAttention(args)
        self.attention = Attention_global()
        self.attn_params = self.attention.parameters()
        
        self.linear = Linear_global(feature_num=self.args.output_size)
        self.linear_params = self.linear.parameters()
        
        self.sketch_embedding_network = eval(args.backbone_name + "(args)")
        self.sketch_attention = Attention_global()
        # self.sketch_attention = SelfAttention(args)
        self.sketch_linear = Linear_global(feature_num=self.args.output_size)
        

        if self.args.use_kaiming_init:
            self.attention.apply(init_weights)
            self.linear.apply(init_weights)
            
            self.sketch_attention.apply(init_weights)
            self.sketch_linear.apply(init_weights)
            
        self.optimizer = optim.AdamW([
            {'params': self.sample_embedding_network.parameters(), 'lr': args.learning_rate},
            {'params': self.sketch_embedding_network.parameters(), 'lr': args.learning_rate},
        ])
        
    def forward(self, batch):
        sketch_img = batch['sketch_img'].to(device)
        positive_img = batch['positive_img'].to(device)
        negative_img = batch['negative_img'].to(device)
        
        positive_feature = self.sample_embedding_network(positive_img)
        negative_feature = self.sample_embedding_network(negative_img)
        sketch_feature = self.sample_embedding_network(sketch_img)
        # sketch_feature = self.sketch_embedding_network(sketch_img)
        
        if self.args.use_attention:
            positive_feature = self.attention(positive_feature)
            negative_feature = self.attention(negative_feature)
            sketch_feature = self.attention(sketch_feature)
            # sketch_feature = self.sketch_attention(sketch_feature)
            
        if self.args.use_linear:
            positive_feature = self.linear(positive_feature)
            negative_feature = self.linear(negative_feature)
            sketch_feature = self.linear(sketch_feature)
            # sketch_feature = self.sketch_linear(sketch_feature)
        
        return sketch_feature, positive_feature, negative_feature
    
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        sketch_feature, positive_feature, negative_feature = self.forward(batch)

        loss = self.loss(sketch_feature, positive_feature, negative_feature)
        loss.backward()
        self.optimizer.step()

        return loss.item() 
    

    def test_forward(self, batch):
        # sketch_feature = self.sketch_embedding_network(batch['sketch_img'].to(device))
        sketch_feature = self.sample_embedding_network(batch['sketch_img'].to(device))
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        
        if self.args.use_attention:
            positive_feature = self.attention(positive_feature)
            sketch_feature = self.attention(sketch_feature)
            # sketch_feature = self.sketch_attention(sketch_feature)
        
        if self.args.use_linear:
            positive_feature = self.linear(positive_feature)
            sketch_feature = self.linear(sketch_feature)
            # sketch_feature = self.sketch_linear(sketch_feature)
            
        return sketch_feature, positive_feature
    
    def evaluate(self, datloader_test):
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
        
        self.eval()
        for _, sanpled_batch in enumerate(tqdm(datloader_test)):
            sketch_feature, positive_feature= self.test_forward(sanpled_batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Sketch_Name.extend(sanpled_batch['sketch_path'])

            for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
                if positive_name not in Image_Name:
                    Image_Name.append(sanpled_batch['positive_path'][i_num])
                    Image_Feature_ALL.append(positive_feature[i_num])

        rank = torch.zeros(len(Sketch_Name))
        rank_percentile = torch.zeros(len(Sketch_Name))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)
        
        avererage_area = []
        avererage_area_percentile = []

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            s_name = Sketch_Name[num]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)

            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()
            rank_percentile[num] = (len(distance) - rank[num]) / (len(distance) - 1)
            
            avererage_area.append(1/rank[num].item() if rank[num].item()!=0 else 1)
            avererage_area_percentile.append(rank_percentile[num].item() if rank_percentile[num].item!=0 else 1)

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top5 = rank.le(5).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        meanA = np.mean(avererage_area_percentile)
        meanB = np.mean(avererage_area)
        
        return top1, top5, top10, meanA, meanB