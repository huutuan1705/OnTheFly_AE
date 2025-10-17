import os
import time
import torch
import argparse
import torch.utils.data as data 

from tqdm import tqdm
from baseline_old.dataset import FGSBIR_Dataset
from baseline_old.model import FGSBIR_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Base Fine-Grained SBIR model')
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--backbone_name', type=str, default='InceptionV3', help='VGG16/InceptionV3/ResNet50')
    parsers.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--num_heads', type=int, default=4)
    parsers.add_argument('--root_dir', type=str, default='/kaggle/input/fg-sbir-dataset')
    parsers.add_argument('--backbone_pretrained', type=str, default='./../')
    parsers.add_argument('--load_backbone_pretrained', type=bool, default=False)
    parsers.add_argument('--attention_pretrained', type=str, default='./../')
    parsers.add_argument('--linear_pretrained', type=str, default='./../')
    parsers.add_argument('--pretrained', type=str, default='./../')
    
    parsers.add_argument('--is_train', type=bool, default=True)
    parsers.add_argument('--load_pretrained', type=bool, default=False)
    parsers.add_argument('--train_backbone', type=bool, default=True)
    parsers.add_argument('--use_attention', type=bool, default=True)
    parsers.add_argument('--use_linear', type=bool, default=True)
    parsers.add_argument('--use_kaiming_init', type=bool, default=True)
    
    parsers.add_argument('--batch_size', type=int, default=48)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--step_size', type=int, default=100)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--learning_rate', type=float, default=0.0001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--eval_freq_iter', type=int, default=100)
    parsers.add_argument('--print_freq_iter', type=int, default=1)
    
    args = parsers.parse_args()
    dataloader_train, dataloader_test = get_dataloader(args=args)
    
    model = FGSBIR_Model(args=args)
    model.to(device)
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    
    step_count, top1, top5, top10, meanA, meanB = -1, 0, 0, 0, 0, 0
    
    # scheduler = StepLR(model.optimizer, step_size=args.step_size, gamma=args.gamma)
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        loss = 0
        for _, batch_data in enumerate(tqdm(dataloader_train)):
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)

        # scheduler.step()
        with torch.no_grad():
            model.eval()
            top1_eval, top5_eval, top10_eval, meanA_eval, meanB_eval = model.evaluate(dataloader_test)
            
            if top10_eval > top10:
                top1, top5, top10, meanA, meanB = top1_eval, top5_eval, top10_eval, meanA_eval, meanB_eval
                torch.save(model.state_dict(), "best_model.pth")
                torch.save(
                    {
                        'sample_embedding_network': model.sample_embedding_network.state_dict(),
                        'sketch_embedding_network': model.sample_embedding_network.state_dict(),
                    }, args.dataset_name + '_backbone.pth')
                
                torch.save({'attention': model.attention.state_dict(),
                            'sketch_attention': model.attention.state_dict(),
                            }, args.dataset_name + '_attention.pth')
                torch.save({'linear': model.linear.state_dict(),
                            'sketch_linear': model.linear.state_dict(),
                            }, args.dataset_name + '_linear.pth')
                
            torch.save(model.state_dict(), "last_model.pth")
            
        # Load model
        # model = FGSBIR_Model(args)
        # model.load_state_dict(torch.load(f"{args.backbone_name}_{args.dataset_name}_best.pth"))
        
        print('Top 1 accuracy:  {:.4f}'.format(top1_eval))
        print('Top 5 accuracy:  {:.4f}'.format(top5_eval))
        print('Top 10 accuracy: {:.4f}'.format(top10_eval))
        print('Mean A:          {:.4f}'.format(meanA_eval))
        print('Mean B:          {:.4f}'.format(meanB_eval))
        print('Loss:            {:.4f}'.format(loss))
        print("================================================")