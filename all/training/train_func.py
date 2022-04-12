import torch
# from tqdm import tqdm

from utils.utils import AverageMeter


def train_iter(args, epoch, model, criterion, optimizer, scheduler, dataloader, logger):
    logger.log(f"TRAINING EPOCH {epoch}...")
    print(f"TRAINING EPOCH {epoch}...")

    device = args.device

    #! define loss dictionary
    log_dict = dict()
    log_dict["loss CE"] = AverageMeter()

    lowest_acc = 1
    highest_acc = 0

    # with tqdm(total=len(dataloader)) as _tqdm:
    #     _tqdm.set_description('epoch: {}/{}'.format(epoch+1, args.num_epochs))
        
    for i, (inputs, targets) in enumerate(dataloader):
        '''
            in default setting, inputs: [4, 16, 3, 256, 256]
        '''

        #! prepare data
        inputs = inputs.to(device)
        if args.model_name != 'estmm':
            targets = targets.squeeze(dim=1).type(torch.LongTensor)
        targets = targets.to(device)
        
        #! model forward
        outs = model(inputs)
        outs = outs.to(device)

        #! compute loss
        loss = criterion(outs, targets)
        
        #! training combo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        #! save results
        batch_size = inputs.shape[0]
        log_dict["loss CE"].update(loss.item(), batch_size)

        #! evaluation for batch
        if (i+1) % 10 == 0:
            with torch.no_grad():
                _, prediction = torch.max(outs.cpu(), 1)
                targets = targets.to('cpu')
                accuracy = torch.sum(prediction == targets.data)/len(targets.data)
                if accuracy > highest_acc:
                    highest_acc = accuracy.item()
                if accuracy < lowest_acc:
                    lowest_acc = accuracy.item()

            # _tqdm.set_postfix(lowest_acc='{:.3f}'.format(lowest_acc), highest_acc='{:.3f}'.format(highest_acc), acc='{:.3f}'.format(accuracy), loss='{:.3f}'.format(loss))
            # _tqdm.update(1)

    logger.log(f"EPOCH {epoch} DONE! [loss CE average: %f] [highest accuracy: %f] [lowest accuracy: %f]\n" % 
               (log_dict["loss CE"].average, highest_acc, lowest_acc))
    print(f"EPOCH {epoch} DONE! [loss CE average: %f] [highest accuracy: %f] [lowest accuracy: %f]\n" % 
               (log_dict["loss CE"].average, highest_acc, lowest_acc))
    logger.scalar_summary('train/loss CE', log_dict['loss CE'].average, epoch)
    logger.scalar_summary('train/highest accuracy', highest_acc, epoch)
    logger.scalar_summary('train/lowest accuracy', lowest_acc, epoch)

