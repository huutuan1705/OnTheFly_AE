import torch.nn as nn

def loss_fn(args, features):
    sketch_features_1 = features['sketch_features_1']
    positive_feature_1 = features['positive_feature_1']
    negative_feature_1 = features['negative_feature_1']
    
    sketch_features_2 = features['sketch_features_2']
    positive_feature_2 = features['positive_feature_2']
    negative_feature_2 = features['negative_feature_2']
    
    triplet_1 = nn.TripletMarginLoss(margin=args.margin)
    triplet_2 = nn.TripletMarginLoss(margin=args.margin)
    
    