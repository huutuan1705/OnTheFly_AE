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
    
def get_transform(type, aug_mode='geometric_strong'):
    """
    Get transform for SimCLR with alternating augmentation modes
    
    Args:
        type: 'train' or 'val'/'test'
        aug_mode: 'default', 'geometric_strong', 'color_strong'
            - 'default': balanced augmentation
            - 'geometric_strong': strong geometric, weak color
            - 'color_strong': strong color, weak geometric
    """
    
    if type == 'train':
        if aug_mode == 'geometric_strong':
            # Strong geometric augmentation, weak color augmentation
            transform_list = [
                transforms.RandomResizedCrop(299, scale=(0.7, 1.0)),  # Stronger crop
                transforms.RandomHorizontalFlip(0.7),  # Higher flip probability
                transforms.RandomRotation(25),  # Stronger rotation
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Add affine
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),  # Weak color
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            
        elif aug_mode == 'color_strong':
            # Strong color augmentation, weak geometric augmentation
            transform_list = [
                transforms.RandomResizedCrop(299, scale=(0.9, 1.0)),  # Weaker crop
                transforms.RandomHorizontalFlip(0.3),  # Lower flip probability
                transforms.RandomRotation(5),  # Weaker rotation
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Strong color
                transforms.RandomGrayscale(p=0.2),  # Add grayscale
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            
        else:  # default mode
            # Balanced augmentation (original)
            transform_list = [
                transforms.RandomResizedCrop(299, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        
    else:  # validation/test
        transform_list = [
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    return transforms.Compose(transform_list)
