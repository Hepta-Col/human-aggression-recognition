import torch
import torch.nn as nn
import gc

from dataset import get_dataset, create_dataloader
from common.args import pack_args
from model.model_utils import create_model

torch.backends.cudnn.benchmark = True

def main():
    #! clear garbage
    gc.collect()
    torch.cuda.empty_cache()
    
    #! parsing args
    print("Packing arguments...")
    args = pack_args()
    print("[", args, "]")
    
    #! preparations
    print("Loading dataset...")
    dataset = get_dataset(args)
    print("Loading data loaders...")
    train_loader, val_loader = create_dataloader(
        dataset=dataset, batch_size=args.batch_size, ratio=args.ratio, num_worker=args.num_workers)
    print("Creating model...")
    model = create_model(args)
    model = nn.DataParallel(model)
    model = model.to(args.device)
    print("Loading checkpoint...")
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint)
    model.eval()
    
    #! evaluation
    

if __name__ == '__main__':
    main()
