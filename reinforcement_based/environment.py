import torch 
import os
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from reinforcement_based.reinforcement import Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        train_pickle = os.path.join(args.root_dir, 'train_' + args.dataset_name +'.pickle')
        test_pickle = os.path.join(args.root_dir, 'test_' + args.dataset_name +'.pickle')
        with open(train_pickle, "rb") as f:
            self.Image_Array_Train, self.Sketch_Array_Train, self.Image_Name_Train, self.Sketch_Name_Train = pickle.load(f)
        with open(test_pickle, "rb") as f:
            self.Image_Array_Test, self.Sketch_Array_Test, self.Image_Name_Test, self.Sketch_Name_Test = pickle.load(f)

        self.policy_network = Policy().to(device)
        
    def get_reward(self, action, sketch_name):
        sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
        position_query = self.Image_Name_Train.index(sketch_query_name)
        target_distance = F.pairwise_distance(F.normalize(action),
                                              self.Image_Array_Train[position_query])
        distance = F.pairwise_distance(F.normalize(action), self.Image_Array_Train)
        rank = distance.le(target_distance).sum()

        if rank.item() == 0:
            reward = 1.
        else:
            reward = 1. / rank.item()
        return reward
    
    def evaluate_RL(self, step_stddev=1):
        self.policy_network.eval()
        num_steps = len(self.Sketch_Array_Test[0])
        avererage_area = []
        avererage_area_percentile = []
        mean_rank_ourB = []
        mean_rank_ourA = []
        avererage_ourB = []
        avererage_ourA = []
        exps = np.linspace(1, num_steps, num_steps) / num_steps
        factor = np.exp(1 - exps) / np.e
        sketch_range = []
        
        rank_all = torch.zeros(len(self.Sketch_Array_Test), num_steps)
        rank_all_percentile = torch.zeros(len(self.Sketch_Array_Test), num_steps)
        
        for i_batch, sanpled_batch in enumerate(self.Sketch_Array_Test):
            #print('evaluate_RL running', i_batch)
            sketch_name = self.Sketch_Name_Test[i_batch]
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = self.Image_Name_Test.index(sketch_query_name)
            mean_rank = []
            mean_rank_percentile = []
            
            for i_sketch in range(sanpled_batch.shape[0]):
                _, sketch_feature, _, _  = self.policy_network.select_action(sanpled_batch[i_sketch].unsqueeze(0).to(device))
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


    def calculate_loss(self, log_probs, rewards, entropies):
        loss = 0
        gamma = 0.9
        for i in reversed(range(len(rewards))):
            #R = gamma ** (len(rewards) - i -1) * rewards[i]
            R =  rewards[i] # Flat Reward
            loss = loss - log_probs[i] * R #- 0.0001 * entropies[i]
        loss = loss / len(rewards)
        return loss