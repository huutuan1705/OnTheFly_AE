import torch
import torch.utils.data as data

from baseline.backbones import InceptionV3
from baseline.attention import SelfAttention, Linear_global
from phase2.datasets import FGSBIR_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))

    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))

    return dataloader_train, dataloader_test

def save_features(args, dataloader):
    backbones_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_backbone.pth", weights_only=True)
    attention_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_attention.pth", weights_only=True)
    linear_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_linear.pth", weights_only=True)
    
    sample_network = InceptionV3(args).to(device)
    sample_network.load_state_dict(backbones_state['sample_embedding_network'], strict=False)
    sample_network.fix_weights()
    sample_network.eval()
    
    sketch_network = InceptionV3(args).to(device)
    sketch_network.load_state_dict(backbones_state['sketch_embedding_network'], strict=False)
    sketch_network.fix_weights()
    sketch_network.eval()
    
    sample_attention = SelfAttention(args).to(device)
    sample_attention.load_state_dict(attention_state['attention'], strict=False)
    sample_attention.fix_weights()
    sample_attention.eval()
    
    sketch_attention = SelfAttention(args).to(device)
    sketch_attention.load_state_dict(attention_state['sketch_attention'], strict=False)
    sketch_attention.fix_weights()
    sketch_attention.eval()
    
    linear = Linear_global(args).to(device)
    linear.load_state_dict(linear_state['linear'], strict=False)
    linear.fix_weights()
    linear.eval()
    
    image_array = torch.FloatTensor().to(device)
    sketch_features = torch.FloatTensor().to(device)
    sketch_array = []
    image_names = []
    sketch_name = []
    
    for i_batch, sample_batch in enumerate(dataloader):
        pass