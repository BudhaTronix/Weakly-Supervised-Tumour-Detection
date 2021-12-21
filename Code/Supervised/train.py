import copy
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Code.Utils.loss import DiceLoss

scaler = GradScaler()
torch.set_num_threads(1)


def saveImage(img, lbl, op, disp_imgs):
    # create grid of images
    ctr = 1
    figure = plt.figure(figsize=(10, 10))
    for i in range(0, disp_imgs):
        plt.subplot(3, disp_imgs, ctr, title="MRI")
        plt.axis('off')
        plt.imshow(img[:, :, i], cmap="gray")
        ctr += 1
    for i in range(0, disp_imgs):
        plt.subplot(3, disp_imgs, ctr, title="GT")
        plt.axis('off')
        plt.imshow(lbl[:, :, i], cmap="gray")
        ctr += 1
    for i in range(0, disp_imgs):
        plt.subplot(3, disp_imgs, ctr, title="OP")
        plt.axis('off')
        plt.imshow(op.to(torch.float)[:, :, i], cmap="gray")
        ctr += 1

    return figure


def train(dataloaders, modelPath, modelPath_bestweight, num_epochs, model, optimizer,
          log=False, device="cuda:4"):
    if log:
        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        TBLOGDIR = "/project/mukhopad/tmp/LiverTumorSeg/Code/Supervised/runs/Training/{}".format(start_time)
        writer = SummaryWriter(TBLOGDIR)
    best_model_wts = ""
    best_acc = 0.0
    best_val_loss = 99999
    since = time.time()
    model.to(device)
    params_to_update = model.parameters()
    optimizer = torch.optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
    criterion = DiceLoss()
    store_idx = 1  # int(len(dataloaders[0])/2)
    # criterion = torch.nn.
    disp_imgs = 6
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in [0]:
            if phase == 0:
                print("Model In Training mode")
                model.train()  # Set model to training mode
            else:
                print("Model In Validation mode")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            idx = 0
            # Iterate over data.
            for batch in tqdm(dataloaders):
                image_batch, labels_batch = batch

                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 0):
                    with autocast(enabled=True):
                        outputs = model(
                            image_batch.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float))
                        loss, acc = criterion(outputs[0].squeeze(1).squeeze(0),
                                              labels_batch.permute(2, 0, 1).to(device))

                    # backward + optimize only if in training phase
                    if phase == 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        if epoch % 50 == 0:
                            s = random.randint(1, 21)
                            img = image_batch[:, :, s:(s + disp_imgs)].detach().cpu()
                            lbl = labels_batch[:, :, s:(s + disp_imgs)].detach().cpu()
                            op = outputs[0].squeeze().permute(1, 2, 0)[:, :, s:(s + disp_imgs)].detach().cpu()

                            fig = saveImage(img, lbl, op, disp_imgs)
                            text = "Epoch : " + str(epoch)
                            writer.add_figure(text, fig, epoch)

                    # statistics
                    running_loss += loss.item()
                    """outputs = outputs[0].cpu().detach().numpy() >= threshold_otsu(outputs[0].cpu().detach().numpy())
                    running_corrects += f1_score(outputs.astype(int).flatten(), labels_batch.numpy().flatten(),
                                                 average='macro')"""
                    running_corrects += acc.item()
                    idx += 1

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
            torch.save(model.state_dict(), modelPath)
            # load best model weights
            # model.load_state_dict(best_model_wts)
            # torch.save(model, modelPath_bestweight)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    print("Saving the model")
    # save the model
    torch.save(model, modelPath)
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, modelPath_bestweight)
