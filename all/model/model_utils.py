import torch
import pickle
import os
from matplotlib import pyplot as plt

from model.vivit.vivit import create_vivit
from model.c3d.c3d import create_c3d
from model.estmm.estmm import create_estmm


def create_model(args):
    if args.model_name == 'c3d':
        return create_c3d(args)
    elif 'vivit' in args.model_name:
        return create_vivit(args)
    elif args.model_name == 'estmm':
        assert args.dataset == 'rwf_npy', "If your model is ESTMM, you should use RWF_npy dataset!"
        return create_estmm(args)
    else:
        raise NotImplementedError()


def save_model(model, save_dir):
    torch.save(model.state_dict(), save_dir + 'save.pt')


def get_num_parameter(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load_checkpoint(logdir, mode='last'):
    if mode == 'last':
        model_path = os.path.join(logdir, 'last.model')
        optim_path = os.path.join(logdir, 'last.optim')
        config_path = os.path.join(logdir, 'last.config')
    elif mode == 'best':
        model_path = os.path.join(logdir, 'best.model')
        optim_path = os.path.join(logdir, 'best.optim')
        config_path = os.path.join(logdir, 'best.config')

    else:
        raise NotImplementedError()

    print("=> Loading checkpoint from '{}'".format(logdir))
    if os.path.exists(model_path):
        model_state = torch.load(model_path)
        optim_state = torch.load(optim_path)
        with open(config_path, 'rb') as handle:
            cfg = pickle.load(handle)
    else:
        return None, None, None

    return model_state, optim_state, cfg


def save_checkpoint(epoch, model_state, optim_state, logdir):
    last_model = os.path.join(logdir, 'last.model')
    last_optim = os.path.join(logdir, 'last.optim')
    last_config = os.path.join(logdir, 'last.config')

    opt = {
        'epoch': epoch,
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_model_diagrams(probs, labels, n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    confidences, predictions = probs.max(1)
    accuracies = torch.eq(predictions, labels)
    f, rel_ax = plt.subplots(1, 2, figsize=(4, 2.5))

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in
                   zip(bins[:-1], bins[1:])]
    bin_corrects = [torch.mean(accuracies[bin_index])
                    for bin_index in bin_indices]
    bin_scores = [torch.mean(confidences[bin_index])
                  for bin_index in bin_indices]

    confs = rel_ax.bar(bins[:-1], bin_corrects.numpy(), width=width)
    gaps = rel_ax.bar(bins[:-1], (bin_scores - bin_corrects).numpy(), bottom=bin_corrects.numpy(), color=[1, 0.7, 0.7],
                      alpha=0.5, width=width, hatch='//', edgecolor='r')
    rel_ax.plot([0, 1], [0, 1], '--', color='gray')
    rel_ax.legend([confs, gaps], ['Outputs', 'Gap'],
                  loc='best', fontsize='small')

    # Clean up
    rel_ax.set_ylabel('Accuracy')
    rel_ax.set_xlabel('Confidence')
    f.tight_layout()
    return f
