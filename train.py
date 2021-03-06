import time
import copy
import torch
from torch.utils.tensorboard import SummaryWriter

##Metrics
from sklearn.metrics import f1_score

def train(model, criterion, optimizer, scheduler, dataloaders,device, logger=None, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    val_acc_history = []
    train_loss_history = []
    loss_history = []
    f1_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        predictions = []
        gt = []
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    predictions.extend(preds.cpu().numpy())
                    gt.extend(labels.cpu().numpy())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() /  len(dataloaders[phase].dataset)
            f1 = f1_score(gt,predictions,average='weighted')
            print('{} Loss: {:.4f} Acc: {:.4f}  F1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, f1))
            
            #if phase == 'train':
            #    logger.add_scalar('Loss/train',epoch_loss,epoch)
            #    logger.add_scalar('F1/train',f1,epoch)
                
            #if phase == 'val':
            #    logger.add_scalar('Loss/val',epoch_loss,epoch)
            #    logger.add_scalar('F1/val',f1,epoch)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            # deep copy the model
            if phase == 'val' and f1 > best_f1:
                best_f1 = f1
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                loss_history.append(epoch_loss)
                f1_history.append(f1)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best F1 {:.4f} and val Acc: {:4f}'.format(best_f1, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history,train_loss_history,  loss_history, f1_history
