import os
import torch 
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        train_pickle = os.path.join(args.root_dir, 'train_' + args.dataset_name +'.pickle')
        test_pickle = os.path.join(args.root_dir, 'test_' + args.dataset_name +'.pickle')
        with open(train_pickle, "rb") as f:
            self.Image_Array_Train, self.Sketch_Array_Train, self.Image_Name_Train, self.Sketch_Name_Train = pickle.load(f)
        with open(test_pickle, "rb") as f:
            self.Image_Array_Test, self.Sketch_Array_Test, self.Image_Name_Test, self.Sketch_Name_Test = pickle.load(f)
            
        