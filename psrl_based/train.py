import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, args):
    model = model.to(device)
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained_dir), strict=False)
    
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    optimizer = optim.Adam([
        {'params': model.bilstm.parameters(), 'lr': args.lr},
    ])
    loss_buffer = []

    top5, top10, avg_loss = 0, 0, 0
    for i_epoch in range(args.epochs):
        model.bilstm.train()
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        for i, sanpled_batch in enumerate(tqdm(model.Sketch_Array_Train)):
            loss_step = 0
            loss_triplet = 0
            positive, negative, semantic_vector = model.get_sample(model.Sketch_Name_Train[i])
            sketch_features = model(sanpled_batch, semantic_vector).squeeze(0)
            for i_sketch in range(sanpled_batch.shape[0]):
                sketch_feature = sketch_features[i_sketch].to(device)
                loss_triplet += loss_fn(sketch_feature, positive.to(device), negative.to(device))
                
            loss_step += loss_triplet
            loss_buffer.append(loss_step)
        
            if (i + 1) % 20 == 0:
                optimizer.zero_grad()
                policy_loss = torch.stack(loss_buffer).mean()
                policy_loss.backward()
                utils.clip_grad_norm_(model.bilstm.parameters(), 40)
                optimizer.step()
                loss_buffer = []
    
        with torch.no_grad():
            top1_eval, top5_eval, top10_eval, meanA, meanB, meanOurA, meanOurB  = model.evaluate_lstm()
            model.bilstm.train()
        
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