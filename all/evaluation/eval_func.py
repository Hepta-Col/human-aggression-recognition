import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(args, model, val_loader):
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for i, (inputs, targets) in enumerate(val_loader):
        outs = model(inputs)
        outs = F.softmax(outs, dim=1)
        _, predictions = torch.max(outs, dim=1)
        predictions = predictions.to('cpu')
        if args.model_name != 'estmm':
            targets = targets.squeeze(dim=1)
        targets = targets.to('cpu')

        the_accuracy = accuracy_score(targets, predictions)
        the_precision = precision_score(targets, predictions, zero_division=0)
        the_recall = recall_score(targets, predictions, zero_division=0)
        the_f1 = f1_score(targets, predictions, zero_division=0)

        accuracy_list.append(the_accuracy)
        precision_list.append(the_precision)
        recall_list.append(the_recall)
        f1_list.append(the_f1)

    accuracy_list = torch.Tensor(accuracy_list)
    precision_list = torch.Tensor(precision_list)
    recall_list = torch.Tensor(recall_list)
    f1_list = torch.Tensor(f1_list)

    accuracy = torch.sum(accuracy_list)/len(accuracy_list)
    precision = torch.sum(precision_list)/len(precision_list)
    recall = torch.sum(recall_list)/len(recall_list)
    f1 = torch.sum(f1_list)/len(f1_list)

    return accuracy, precision, recall, f1    

