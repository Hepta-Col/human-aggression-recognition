import torch
from torch import nn
import torch.optim as optim
import gc
from colorama import Fore, Style

from common.args import pack_args
from training.train_func import train_iter
from evaluation.eval_func import *
from model.model_utils import *
from utils.utils import *
from dataset.dataset_utils import get_dataset, create_dataloader

torch.backends.cudnn.benchmark = True


def main():
    #! clear garbage
    gc.collect()
    torch.cuda.empty_cache()

    #! parsing args
    print("Loading arguments...")
    args = pack_args()
    print(f"[{args}]")

    logger = Logger(f"{args.model_name}_on_{args.dataset}")
    logger.log(f"[{args}]")

    #! preparations
    print("Loading dataset...")
    train_dataset = get_dataset(args, train=True)
    print(f"{args.dataset} dataset has {len(train_dataset)} clips in total.")
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloader(
        dataset=train_dataset, batch_size=args.batch_size, ratio=args.ratio, num_worker=args.num_workers)
    print(f"train set loader size: {len(train_loader)}, val set loader size: {len(val_loader)}")
    print("Creating model...")
    model = create_model(args)
    print("model parameters: ", get_num_parameter(model))
    model = nn.DataParallel(model)
    model = model.to(args.device)
    print("Creating optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("using optimizer: ", optimizer)
    print("Creating scheduler...")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.T_max)
    print("using scheduler: ", scheduler)
    print("Creating criterion...")
    criterion = nn.CrossEntropyLoss()
    print("using criterion: ", criterion)

    #! training
    print(Fore.RED + Style.BRIGHT + "Start training!" + Style.RESET_ALL)

    for epoch in range(args.num_epochs):
        model.train()

        kwargs = {}
        kwargs['epoch'] = epoch
        kwargs['model'] = model
        kwargs['criterion'] = criterion
        kwargs['optimizer'] = optimizer
        kwargs['scheduler'] = scheduler
        kwargs['dataloader'] = train_loader
        kwargs['logger'] = logger

        train_iter(args, **kwargs)

        if (epoch + 1) % 10 == 0:
            model.eval()
            print(Fore.YELLOW + Style.BRIGHT + f"evaluating for epoch {epoch}..." + Style.RESET_ALL)
            with torch.no_grad():
                accuracy, precision, recall, f1 = evaluate(args, model, val_loader)
                print(Fore.GREEN + Style.BRIGHT + "logging..." + Style.RESET_ALL)
                logger.log(
                    f"\nFor epoch {epoch}, accuracy={accuracy}, precision={precision}, recall={recall}, f1={f1}\n")
                print(
                    f"\nFor epoch {epoch}, accuracy={accuracy}, precision={precision}, recall={recall}, f1={f1}\n")

                logger.scalar_summary('train/accuracy', accuracy, epoch)
                logger.scalar_summary('train/precision', precision, epoch)
                logger.scalar_summary('train/recall', recall, epoch)
                logger.scalar_summary('train/f1', f1, epoch)

    #! save model
    print("Saving...")
    save_model(model, args.save_path)
    
    print(Fore.RED + Style.BRIGHT + "All done!" + Style.RESET_ALL)


if __name__ == '__main__':
    main()
