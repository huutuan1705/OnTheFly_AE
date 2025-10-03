import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn.utils as utils
from tqdm import tqdm
from torch import optim
from phase2.datasets import FGSBIR_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))

    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))

    return dataloader_train, dataloader_test

def train_model(model, args):
    step_stddev = 1
    model = model.to(device)
    model.policy_network.train()
    dataloader_train, dataloader_test = get_dataloader(args)
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    optimizer = optim.Adam([
        {'params': model.policy_network.parameters(), 'lr': args.lr},
    ])
    loss_buffer = []

    top5, top10, avg_loss = 0, 0, 0
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        for i, sanpled_batch in enumerate(model.Sketch_Array_Train):
            entropies = []
            log_probs = []
            rewards = []
            
            for i_sketch in range(sanpled_batch.shape[0]):
                action_mean, sketch_anchor_embedding, log_prob, entropy = model.policy_network.select_action(sanpled_batch[i_sketch].unsqueeze(0).to(device))
                reward = model.get_reward(sketch_anchor_embedding, model.Sketch_Name_Train[i])
                
                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
            
            loss_single = model.calculate_loss(log_probs, rewards, entropies)
            loss_buffer.append(loss_single)
            
            if (i + 1) % 16 == 0:
                optimizer.zero_grad()
                policy_loss = torch.stack(loss_buffer).mean()
                policy_loss.backward()
                utils.clip_grad_norm_(model.policy_network.parameters(), 40)
                optimizer.step()
                loss_buffer = []
            
            if (i + 1) % 400 == 0:
                with torch.no_grad():
                    top1_eval, top5_eval, top10_eval, meanA, meanB, meanOurA, meanOurB  = model.evaluate_RL()
                    model.policy_network.train()
                
                if top5_eval > top5:
                    top5 = top5_eval
                    torch.save(model.state_dict(), "best_top5_model.pth")
                if top10_eval > top10:
                    top10 = top10_eval
                    torch.save(model.state_dict(), "best_top10_model.pth")
                        
                torch.save(model.state_dict(), "last_model.pth")
                print('Top 1 accuracy : {:.5f}'.format(top1_eval))
                print('Top 5 accuracy : {:.5f}'.format(top5_eval))
                print('Top 10 accuracy: {:.5f}'.format(top10_eval))
                print('Mean A         : {:.5f}'.format(meanA))
                print('Mean B         : {:.5f}'.format(meanB))
                print('meanOurA       : {:.5f}'.format(meanOurA))
                print('meanOurB       : {:.5f}'.format(meanOurB))
                print('Loss           : {:.5f}'.format(avg_loss))        