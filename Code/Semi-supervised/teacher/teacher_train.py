import copy
import time
from datetime import datetime

import torch
from skimage.filters import threshold_otsu
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Code.Utils.loss import DiceLoss

scaler = GradScaler()


def train(dataloaders, modelPath, modelPath_bestweight, num_epochs, model, optimizer,
          log=False, device="cuda"):
    if log:
        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        TBLOGDIR = "runs/Training/Unet/{}".format(start_time)
        writer = SummaryWriter(TBLOGDIR)
    best_model_wts = ""
    best_acc = 0.0
    best_val_loss = 99999
    since = time.time()
    model.to(device)
    criterion = DiceLoss()
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

                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 0):
                    with autocast(enabled=True):
                        image_batch = (image_batch - image_batch.min()) / \
                                      (image_batch.max() - image_batch.min())  # Min Max normalization
                        # image_batch = image_batch / np.linalg.norm(image_batch)  # Gaussian Normalization
                        outputs = model(image_batch.to(device))
                        # outputs = (outputs - outputs.min())/(outputs.max() - outputs.min())  # Min Max normalization
                        loss = criterion(outputs, labels_batch.to(device))

                    # backward + optimize only if in training phase
                    if phase == 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    # statistics
                    running_loss += loss.item()
                    outputs = outputs.cpu().detach().numpy() >= threshold_otsu(outputs.cpu().detach().numpy())
                    running_corrects += f1_score(outputs.astype(int).flatten(), labels_batch.numpy().flatten(),
                                                 average='macro')

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase])
            if phase == 0:
                mode = "Train"
                if log:
                    writer.add_scalar("Loss/Train", epoch_loss, epoch)
                    writer.add_scalar("Acc/Train", epoch_acc, epoch)
            else:
                mode = "Val"
                if log:
                    writer.add_scalar("Loss/Validation", epoch_loss, epoch)
                    writer.add_scalar("Acc/Validation", epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 1 and (epoch_acc > best_acc or epoch_loss < best_val_loss):
                print("Saving the best model weights")
                best_val_loss = epoch_loss
                best_acc = epoch_acc
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
