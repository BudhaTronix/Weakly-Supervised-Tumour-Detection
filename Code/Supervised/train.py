from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import torch.nn as nn
import copy
import torch
from datetime import datetime, time


def train(dataloaders, modelPath, modelPath_bestweight, num_epochs, model, criterion, optimizer,
          log=False, device="cuda"):
    if log:
        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        TBLOGDIR = "runs/BlurDetection/Training/RenseNet101_SSIM/{}".format(start_time)
        writer = SummaryWriter(TBLOGDIR)
    best_model_wts = ""
    best_acc = 0.0
    best_val_loss = 99999
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in [0, 1]:
            if phase == 0:
                print("Model In Training mode")
                model.train()  # Set model to training mode
            else:
                print("Model In Validation mode")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for batch in tqdm(dataloaders[phase]):
                image_batch, labels_batch = batch
                image_batch = image_batch.unsqueeze(1)

                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 0):
                    with autocast(enabled=True):
                        image_batch = (image_batch - image_batch.min()) / \
                                      (image_batch.max() - image_batch.min())  # Min Max normalization
                        # image_batch = image_batch / np.linalg.norm(image_batch)  # Gaussian Normalization
                        outputs = model(image_batch.float().to(device))
                        loss = criterion(outputs.squeeze(1).float(), labels_batch.float().to(device))

                    # backward + optimize only if in training phase
                    if phase == 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += np.sum(np.around(outputs.detach().cpu().squeeze().numpy(),
                                                         decimals=precision) == np.around(labels_batch.numpy(),
                                                                                          decimals=precision))

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase])

            if phase == 0:
                mode = "Train"
                # train_loss_history.append(epoch_acc.data)
                if log:
                    writer.add_scalar("Loss/Train", epoch_loss, epoch)
                    writer.add_scalar("Acc/Train", epoch_acc, epoch)
            else:
                mode = "Val"
                # val_acc_history.append(epoch_acc.data)
                if log:
                    writer.add_scalar("Loss/Validation", epoch_loss, epoch)
                    writer.add_scalar("Acc/Validation", epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 1 and (epoch_acc.item() >= best_acc or epoch_loss < best_val_loss):
                print("Saving the best model weights")
                best_acc = epoch_acc.item()
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print("Saving the model")
            # save the model
            torch.save(model, modelPath)
            # load best model weights
            model.load_state_dict(best_model_wts)
            torch.save(model, modelPath_bestweight)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    print("Saving the model")
    # save the model
    torch.save(model, modelPath)
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, modelPath_bestweight)