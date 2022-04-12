import argparse


def pack_args():
    parser = argparse.ArgumentParser(description='Train human-action-recognition models')
    
    # model hyperparameters
    parser.add_argument('--model_name', type=str, choices=['c3d', 'vivit_2', 'vivit_3', 'estmm'], default='vivit_3')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_frames', type=int, default=16)
    
    parser.add_argument('--depth', type=int, default=6)
    
    # dataset hyperparameters
    parser.add_argument('--dataset', type=str, choices=[
                        'youtube_small', 'fight_surv', 'rlv', 'rwf', 'rwf_npy'], default='rlv')
    # dataloader hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ratio', type=float, default=0.85)
    parser.add_argument('--num_workers', type=int, default=8)
    # training loop hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--T_max', type=float, default=5.0)
    parser.add_argument('--save_path', type=str, default='/home/qzt/all/save_dir/')
    parser.add_argument('--device', type=str,
                        choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--num_epochs', type=int, default=20)
    # evaluation hyperparameters
    parser.add_argument('--load_path', type=str, default='/home/qzt/all/save_dir/')
    
    return parser.parse_args()


