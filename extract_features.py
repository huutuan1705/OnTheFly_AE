import torch
import torch.utils.data as data
import pickle
import argparse
from tqdm import tqdm

from baseline.backbones import InceptionV3
from baseline.attention import SelfAttention, Linear_global
from phase2.datasets import FGSBIR_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Environtment():
    def __init__(self, args):
        backbones_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_backbone.pth", weights_only=True)
        attention_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_attention.pth", weights_only=True)
        linear_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_linear.pth", weights_only=True)

        sample_embedding_network = InceptionV3(args)
        sample_embedding_network.to(device)
        sample_embedding_network.load_state_dict(backbones_state['sample_embedding_network'], strict=False)
        sample_embedding_network.eval()
        
        sketch_embedding_network = InceptionV3(args)
        sketch_embedding_network.to(device)
        sketch_embedding_network.load_state_dict(backbones_state['sketch_embedding_network'], strict=False)
        sketch_embedding_network.eval()
        
        attention = SelfAttention(args)
        attention.to(device)
        attention.load_state_dict(attention_state['attention'], strict=False)
        attention.eval()
        
        sketch_attention = SelfAttention(args)
        sketch_attention.to(device)
        sketch_attention.load_state_dict(attention_state['sketch_attention'], strict=False)
        sketch_attention.eval()
        
        linear = Linear_global(feature_num=64)
        linear.to(device)
        linear.load_state_dict(linear_state['linear'])
        
        dataset_train = FGSBIR_Dataset(args, mode='train')
        dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))

        self.image_array_train = torch.FloatTensor().to(device)
        self.sketch_array_train = []
        self.image_name_train = []
        self.sketch_name_train = []
        
        for i_batch, sample_batch in enumerate(tqdm(dataloader_train)):
            sketch_feature_all = torch.FloatTensor().to(device)
            for data_sketch in sample_batch['sketch_imgs']:
                sketch_feature = sketch_attention(sketch_embedding_network(data_sketch.to(device)))
                sketch_feature_all = torch.cat((sketch_feature_all, sketch_feature.detach()))
            self.sketch_name_train.extend(sample_batch["sketch_path"])
            self.sketch_array_train.append(sketch_feature_all)
            
            if sample_batch['positive_path'][0] not in self.image_name_train:
                rgb_feature = linear(attention(sample_embedding_network(sample_batch['positive_img'])))
                self.image_array_train = torch.cat((self.image_array_train, rgb_feature.detach()))
                self.image_name_train.extend(sample_batch['positive_path'])
                
        dataset_test = FGSBIR_Dataset(args, mode='test')
        dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))

        self.image_array_test = torch.FloatTensor().to(device)
        self.sketch_array_test = []
        self.image_name_test = []
        self.sketch_name_test = []
        
        for i_batch, sample_batch in enumerate(tqdm(dataloader_test)):
            sketch_feature_all = torch.FloatTensor().to(device)       
            for data_sketch in sample_batch['sketch_imgs']:
                sketch_feature = sketch_attention(sketch_embedding_network(data_sketch.to(device)))
                sketch_feature_all = torch.cat((sketch_feature_all, sketch_feature.detach()))
            self.sketch_name_test.extend(sample_batch["sketch_path"])
            self.sketch_array_test.append(sketch_feature_all)
            
            if sample_batch['positive_path'][0] not in self.image_name_test:
                rgb_feature = linear(attention(sample_embedding_network(sample_batch['positive_img'])))
                self.image_array_test = torch.cat((self.image_array_test, rgb_feature.detach()))
                self.image_name_test.extend(sample_batch['positive_path'])
        
        train_pickle = "train_" + args.dataset_name + ".pickle"    
        test_pickle = "test_" + args.dataset_name + ".pickle"    
        with open(train_pickle, "wb") as f:
            pickle.dump((self.image_array_train, self.sketch_array_train, self.image_name_train, self.sketch_name_train), f)
        print("Extract training feature done")
        
        with open(test_pickle, "wb") as f:
            pickle.dump((self.image_array_test, self.sketch_array_test, self.image_name_test, self.sketch_name_test), f)
        print("Extract testing feature done")
        
if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Baseline Fine-Grained SBIR model')
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--pretrained_dir', type=str, default='')
    parsers.add_argument('--root_dir', type=str, default='/kaggle/input/fg-sbir-dataset')
    parsers.add_argument('--batch_size', type=int, default=1)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--num_heads', type=int, default=8)
    
    args = parsers.parse_args()
    envi = Environtment(args)