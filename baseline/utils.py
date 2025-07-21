import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def np2th(weights, conv=False):
    "Convert HWIO to OIHW"
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def loss_fn(args, sketch_feature, positive_feature, negative_feature, fm_6bs):
    triplet = nn.TripletMarginLoss(margin=args.margin)
    triplet_loss = triplet(sketch_feature, positive_feature, negative_feature)
    mse_loss = F.mse_loss(input=fm_6bs["fm_6b_ske"], target=fm_6bs["fm_6b_pos"], reduction="none")
    
    total_loss = triplet_loss + mse_loss
    total_loss = torch.mean(total_loss)
    return total_loss
    
def get_transform(type):
    if type == 'train':
        transform_list = [
            transforms.RandomResizedCrop(299, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    else: 
        transform_list = [
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    return transforms.Compose(transform_list)