import torch.nn as nn

def loss_fn(args, features):
    sketch_features_1 = features['sketch_features_1'] # shape [N, B, D]
    positive_feature_1 = features['positive_feature_1'].unsqueeze(1) # shape [N, 1, D]
    negative_feature_1 = features['negative_feature_1'].unsqueeze(1)
    
    sketch_features_2 = features['sketch_features_2']
    positive_feature_2 = features['positive_feature_2'].unsqueeze(1)
    negative_feature_2 = features['negative_feature_2'].unsqueeze(1)
    
    criterion = nn.TripletMarginLoss(margin=args.margin)
    loss_triplet_1 = 0
    loss_triplet_2 = 0
    
    for i_batch in range(args.batch_size):
        for i_sketch in range(len(sketch_features_1[i_batch])):
            sub_sketch_feature_1 = sketch_features_1[i_batch][i_sketch] # shape [1, D]
            sub_sketch_feature_2 = sketch_features_2[i_batch][i_sketch] # shape [1, D]
            
            loss_triplet_1 += criterion(sub_sketch_feature_2, positive_feature_1[i_batch], negative_feature_1[i_batch])
            loss_triplet_2 += criterion(sub_sketch_feature_1, positive_feature_2[i_batch], negative_feature_2[i_batch])
    
    total_loss = loss_triplet_1 + loss_triplet_2
    return total_loss        