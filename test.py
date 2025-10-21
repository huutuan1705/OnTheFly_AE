import torch
import argparse

from phase2.model import Siamese_SBIR

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Baseline Fine-Grained SBIR model')
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--num_heads', type=int, default=8)
    parsers.add_argument('--root_dir', type=str, default='/kaggle/input/fg-sbir-dataset')
    parsers.add_argument('--pretrained_dir', type=str, default='/kaggle/input/base_ae_model/pytorch/default/1/best_model.pth')
    parsers.add_argument('--save_dir', type=str, default='/kaggle/working/')
    
    parsers.add_argument('--use_kaiming_init', type=bool, default=True)
    parsers.add_argument('--load_pretrained', type=bool, default=False)
    parsers.add_argument('--stage2', type=bool, default=False)
    
    parsers.add_argument('--batch_size', type=int, default=48)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--step_size', type=int, default=100)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--alpha', type=float, default=0.4)
    parsers.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.0001)
    parsers.add_argument('--epochs', type=int, default=200)
    args = parsers.parse_args()
    
    model = Siamese_SBIR(args).to("cpu")
    old_model = Siamese_SBIR(args).to("cpu")
    
    model_path = "chair_phase2_8669_9319.pth"
    checkpoint = torch.load(model_path, map_location="cpu")
    old_model.load_state_dict(checkpoint)
    
    model.sample_embedding_network.load_state_dict(old_model.sample_embedding_network.state_dict())
    model.sketch_embedding_network.load_state_dict(old_model.sketch_embedding_network.state_dict())
    model.attention.load_state_dict(old_model.attention.state_dict())
    model.sketch_attention.load_state_dict(old_model.sketch_attention.state_dict())
    model.linear.load_state_dict(old_model.linear.state_dict())
    model.attn.norm.load_state_dict(old_model.bilstm.attn.norm.state_dict())
    model.attn.mha.load_state_dict(old_model.bilstm.attn.mha.state_dict())
    model.attn.proj.load_state_dict(old_model.bilstm.proj.state_dict())
    
    torch.save(model.state_dict(), "chair_model.pth")