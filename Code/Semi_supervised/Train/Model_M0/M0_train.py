import time
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Code.Utils.loss import DiceLoss

torch.set_num_threads(1)
scaler = GradScaler()


def saveImage(img, lbl, op):
    # create grid of images
    figure = plt.figure(figsize=(10, 10))
    plt.subplot(131, title="MRI")
    plt.grid(False)
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.subplot(132, title="GT")
    plt.grid(False)
    plt.imshow(lbl.permute(1, 2, 0), cmap="gray")
    plt.subplot(133, title="OP")
    plt.grid(False)
    plt.imshow(op.permute(1, 2, 0).to(torch.float), cmap="gray")

    return figure


def train(dataloaders, modelPath, modelPath_bestweight, num_epochs, model, optimizer, device="cuda",
          log=False, logPath=""):
    if log:
        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        TBLOGDIR = logPath + "{}".format(start_time)
        writer = SummaryWriter(TBLOGDIR)
    best_acc = 0.0
    best_val_loss = 99999
    since = time.time()
    model.to(device)
    criterion = DiceLoss()
    store_idx = int(len(dataloaders[0]) / 2)
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
            idx = 0
            # Iterate over data.
            for batch in tqdm(dataloaders[phase]):
                image_batch, labels_batch = batch

                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 0):
                    with autocast(enabled=True):
                        outputs = model(image_batch.unsqueeze(1).to(device))
                        loss, acc = criterion(outputs[0].squeeze(1), labels_batch.to(device))

                    # backward + optimize only if in training phase
                    if phase == 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        if epoch % 5 == 0 and idx == store_idx and log:
                            temp = labels_batch.squeeze().detach().cpu()
                            slice = 0
                            for i in range(len(temp)):
                                if temp[i].max() == 1:
                                    slice = i
                                    break
                            # print("Storing images", idx, epoch)
                            img = image_batch.squeeze()[slice, :, :].unsqueeze(0)
                            lbl = labels_batch.squeeze()[slice, :, :].unsqueeze(0)
                            op = outputs.squeeze()[slice, :, :].unsqueeze(0).cpu().detach()

                            fig = saveImage(img, lbl, op)
                            text = "Epoch : " + str(epoch)
                            writer.add_figure(text, fig, epoch)

                    # statistics
                    running_loss += loss.item()
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

            logging.info('Epoch: {} Mode: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, mode, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 1 and (epoch_acc > best_acc or epoch_loss < best_val_loss):
                logging.info("Saving the best model weights of M0")
                best_val_loss = epoch_loss
                best_acc = epoch_acc
                torch.save(model.state_dict(), modelPath_bestweight)

        # save the model weights after an interval
        if epoch % 10 == 0:
            logging.info("Saving M0 model weights")
            torch.save(model.state_dict(), modelPath)

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # save the model
    logging.info("Saving M0 model weights before exiting")
    torch.save(model.state_dict(), modelPath)

    logging.info("############################# END M0 Model Training #############################")
