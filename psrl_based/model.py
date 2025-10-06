import os
import torch 
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from baseline.attention import Linear_global

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM_Module(nn.Module):
    def __init__(self, in_feature):
        super(BiLSTM_Module, self).__init__()
        self.bilstm1 = nn.LSTM(input_size=in_feature, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.bilstm2 = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = Linear_global(feature_num=64)
        
    def forward(self, x):
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)
        x = self.proj(x)
        
        return x
    
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        train_pickle = os.path.join(args.pickle_dir, 'train_' + args.dataset_name +'.pickle')
        test_pickle = os.path.join(args.pickle_dir, 'test_' + args.dataset_name +'.pickle')
        with open(train_pickle, "rb") as f:
            self.Image_Array_Train, self.Sketch_Array_Train, self.Image_Name_Train, self.Sketch_Name_Train = pickle.load(f)
        with open(test_pickle, "rb") as f:
            self.Image_Array_Test, self.Sketch_Array_Test, self.Image_Name_Test, self.Sketch_Name_Test = pickle.load(f)
        
        csv_path = os.path.join(args.root_dir, args.dataset_name, args.dataset_name + '_labels.csv')
        self.semantic_df = pd.read_csv(csv_path)
        self.semantic_dim = self.semantic_df.shape[1] - 1
        
        self.bilstm = BiLSTM_Module(in_feature=2048+self.semantic_dim).to(device)
        
    def forward(self, x, semantic_vector):
        semantic_expand = semantic_vector.unsqueeze(0).repeat(x.size(0), 1) #(20, 6)
        emb_vector = torch.cat([x.to(device), semantic_expand.to(device)], dim=1) 
        output = self.bilstm(emb_vector.unsqueeze(0))
        
        return output
    
    def get_sample(self, sketch_name):
        sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])

        position_query = self.Image_Name_Train.index(sketch_query_name)
        positive = self.Image_Array_Train[position_query]

        negative_index = position_query
        while(negative_index == position_query):
            negative_index = np.random.randint(0, len(self.Image_Array_Train))

        negative = self.Image_Array_Train[negative_index]
        
        image_file_name = sketch_query_name + ".png"
        row = self.semantic_df[self.semantic_df['image_name'] == image_file_name]
        if row.empty:
            semantic_vec = torch.zeros(len(self.semantic_df.columns) - 1)
        else:
            attr_values = row.drop(columns=['image_name']).iloc[0].to_numpy(dtype=np.float32)
            semantic_vec = torch.tensor(attr_values)
            
        return positive, negative, semantic_vec
    
    def evaluate_lstm(self):
        self.bilstm.eval()
        num_steps = len(self.Sketch_Array_Test[0])
        avererage_area = []
        avererage_area_percentile = []
        mean_rank_ourB = []
        mean_rank_ourA = []
        avererage_ourB = []
        avererage_ourA = []
        exps = np.linspace(1, num_steps, num_steps) / num_steps
        factor = np.exp(1 - exps) / np.e
        
        rank_all = torch.zeros(len(self.Sketch_Array_Test), num_steps)
        rank_all_percentile = torch.zeros(len(self.Sketch_Array_Test), num_steps)
        
        for i_batch, sanpled_batch in enumerate(tqdm(self.Sketch_Array_Test)):
            #print('evaluate_RL running', i_batch)
            sketch_name = self.Sketch_Name_Test[i_batch]
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = self.Image_Name_Test.index(sketch_query_name)
            mean_rank = []
            mean_rank_percentile = []
            sketch_features = self.bilstm(sanpled_batch.unsqueeze(0)).squeeze(0)
            
            for i_sketch in range(sanpled_batch.shape[0]):
                sketch_feature = sketch_features[i_sketch].unsqueeze(0).to(device)
                target_distance = F.pairwise_distance(F.normalize(sketch_feature), self.Image_Array_Test[position_query].unsqueeze(0))
                distance = F.pairwise_distance(F.normalize(sketch_feature), self.Image_Array_Test)
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()
                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                
                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
                    mean_rank_ourB.append(1/rank_all[i_batch, i_sketch].item() * factor[i_sketch])
                    mean_rank_ourA.append(rank_all_percentile[i_batch, i_sketch].item()*factor[i_sketch])
                    
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
            avererage_ourB.append(np.sum(mean_rank_ourB)/len(mean_rank_ourB))
            avererage_ourA.append(np.sum(mean_rank_ourA)/len(mean_rank_ourA))
            
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        
        meanMA = np.mean(avererage_area_percentile)
        meanMB = np.mean(avererage_area)
        meanOurB = np.mean(avererage_ourB)
        meanOurA = np.mean(avererage_ourA)
        
        return top1_accuracy, top5_accuracy, top10_accuracy, meanMA, meanMB, meanOurA, meanOurB