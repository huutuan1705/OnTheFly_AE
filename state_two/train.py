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