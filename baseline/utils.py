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

def info_nce_loss(args, features_view1, features_view2):
        temperature = args.temperature
        batch_size = features_view1.shape[0]

        features = torch.cat([features_view1, features_view2], dim=0)  # [2B, D]

        similarity_matrix = torch.matmul(features, features.T)  # [2B, 2B]

        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(device)

        # Mask self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        logits = similarity_matrix / temperature

        loss = F.cross_entropy(logits, labels)
        return loss
    
    
def loss_fn(args, features):
    sketch_feature_1 = features['sketch_feature_1']
    positive_feature_1 = features['positive_feature_1']
    negative_feature_1 = features['negative_feature_1']
    fm_6bs_1 = features['fm_6bs_1']
    
    sketch_feature_2 = features['sketch_feature_2']
    positive_feature_2 = features['positive_feature_2']
    negative_feature_2 = features['negative_feature_2']
    fm_6bs_2 = features['fm_6bs_2']
    
    triplet_1 = nn.TripletMarginLoss(margin=args.margin)
    triplet_loss_1 = triplet_1(sketch_feature_2, positive_feature_1, negative_feature_1)
    mse_loss_1 = F.mse_loss(input=fm_6bs_1["fm_6b_ske"], target=fm_6bs_1["fm_6b_pos"], reduction="none")
    
    triplet_2 = nn.TripletMarginLoss(margin=args.margin)
    triplet_loss_2 = triplet_2(sketch_feature_1, positive_feature_2, negative_feature_2)
    mse_loss_2 = F.mse_loss(input=fm_6bs_2["fm_6b_ske"], target=fm_6bs_2["fm_6b_pos"], reduction="none")
    
    infonce_sketch = info_nce_loss(args, sketch_feature_1, sketch_feature_2)
    infonce_positive = info_nce_loss(args, positive_feature_1, positive_feature_2)
    
    total_loss = triplet_loss_1 + triplet_loss_2 + 0.2*mse_loss_1 + 0.2*mse_loss_2 + 0.1*infonce_positive + 0.1*infonce_sketch
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
                transforms.RandomResizedCrop(299, scale=(0.6, 1.0)),  # Stronger crop
                transforms.RandomHorizontalFlip(0.7),  # Higher flip probability
                transforms.RandomRotation(50),  # Stronger rotation
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Add affine
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),  # Weak color
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            
        elif aug_mode == 'color_strong':
            # Strong color augmentation, weak geometric augmentation
            transform_list = [
                transforms.RandomResizedCrop(299, scale=(0.9, 1.0)),  # Weaker crop
                transforms.RandomHorizontalFlip(0.5),  # Lower flip probability
                transforms.RandomRotation(5),  # Weaker rotation
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Strong color
                transforms.RandomGrayscale(p=0.7),  # Add grayscale
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.7, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
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
