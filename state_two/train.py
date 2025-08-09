import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from state_two.datasets import FGSBIR_Dataset
from state_two.losses import loss_fn
from baseline.train import evaluate_model

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
    model = model.to(device)
    dataloader_train, dataloader_test = get_dataloader(args)
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained_dir))

    optimizer = optim.Adam(params=model.sketch_linear.parameters(), lr=args.lr)
    top1, top5, top10, avg_loss = 0, 0, 0, 0
    
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")

        losses = []
        for _, batch_data in enumerate(tqdm(dataloader_train, dynamic_ncols=False)):
            model.train()
            optimizer.zero_grad()

            features = model(batch_data)
            loss = loss_fn(args, features)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        top1_eval, top5_eval, top10_eval, meanA, meanB = evaluate_model(
            model, dataloader_test)

        if top5_eval > top5:
            top5 = top5_eval
            torch.save(model.state_dict(), "best_top5_model.pth")

        if top10_eval > top10:
            top10 = top10_eval
            torch.save(model.state_dict(), "best_top10_model.pth")

        print('Top 1 accuracy:  {:.4f}'.format(top1_eval))
        print('Top 5 accuracy:  {:.4f}'.format(top5_eval))
        print('Top 10 accuracy: {:.4f}'.format(top10_eval))
        print('Mean A         : {:.4f}'.format(meanA))
        print('Mean B         : {:.4f}'.format(meanB))
        print('Loss:            {:.4f}'.format(avg_loss))
        with open("results_log.txt", "a") as f:
            f.write("Epoch {:d} | Top1: {:.4f} | Top5: {:.4f} | Top10: {:.4f} | MeanA: {:.4f} | MeanB: {:.4f} | Loss: {:.4f}\n".format(
                i_epoch+1, top1_eval, top5_eval, top10_eval, meanA, meanB, avg_loss))