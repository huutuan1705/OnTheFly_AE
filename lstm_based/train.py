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

def train_model(model, args):
    model = model.to(device)
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained_dir), strict=False)
    
    model.train()
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    optimizer = optim.Adam([
        {'params': model.bilstm.parameters(), 'lr': args.lr},
    ])
    loss_buffer = []

    top5, top10, avg_loss = 0, 0, 0
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        for i, sanpled_batch in enumerate(model.Sketch_Array_Train):
            for i_sketch in range(sanpled_batch.shape[0]):
                pass