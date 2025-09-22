import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def info_nce_loss(args, features_view1: torch.Tensor, features_view2: torch.Tensor):
    """
    InfoNCE (NT-Xent) for SimCLR
    features_view1, features_view2: (B, D)
    """
    temperature = float(args.temperature)
    B, D = features_view1.shape
    device = features_view1.device

    z = torch.cat([features_view1, features_view2], dim=0)

    logits = z @ z.t()                              # (2B, 2B)
    mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask, float('-inf'))

    logits = logits / temperature

    labels = torch.cat([
        torch.arange(B, 2*B, device=device),
        torch.arange(0, B, device=device)
    ], dim=0).long()

    loss = F.cross_entropy(logits, labels)
    return loss
    
    
def loss_fn(args, features):
    sketch_feature_1 = features['sketch_feature_1']
    positive_feature_1 = features['positive_feature_1']
    negative_feature_1 = features['negative_feature_1']
    
    criterion = nn.TripletMarginLoss(margin=args.margin)
    
    infonce_cross = info_nce_loss(args=args, features_view1=sketch_feature_1, features_view2=positive_feature_1)
    triplet_loss = criterion(sketch_feature_1, positive_feature_1, negative_feature_1)
    
    total_loss = args.alpha*triplet_loss 
    if args.use_info:
        total_loss = total_loss + args.beta*infonce_cross   
    return total_loss