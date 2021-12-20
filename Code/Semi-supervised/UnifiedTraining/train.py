import copy
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_num_threads(1)
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Code.Utils.loss import DiceLoss

scaler = GradScaler()


def saveImage(mri, mri_op, mri_lbl, ct, ct_op, op):
    # create grid of images
    figure = plt.figure(figsize=(10, 10))

    plt.subplot(231, title="MRI")
    plt.grid(False)
    plt.imshow(mri.permute(1, 2, 0), cmap="gray")

    plt.subplot(232, title="MRI OP")
    plt.grid(False)
    plt.imshow(mri_op.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(233, title="MRI LBL")
    plt.grid(False)
    plt.imshow(mri_lbl.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(234, title="CT")
    plt.grid(False)
    plt.imshow(ct.permute(1, 2, 0), cmap="gray")

    plt.subplot(235, title="CT OP")
    plt.grid(False)
    plt.imshow(ct_op.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(236, title="CT LBL")
    plt.grid(False)
    plt.imshow(torch.tensor(op).permute(1, 2, 0), cmap="gray")

    return figure


def train(dataloaders, M1_model_path, M1_bw_path, M2_model_path, M2_bw_path, num_epochs, modelM0, modelM1, modelM2,
          optimizer, log=False, logPath=""):
    if log:
        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        TBLOGDIR = logPath + "{}".format(start_time)
        writer = SummaryWriter(TBLOGDIR)

    GPU_ID_M0 = "cuda:5"
    GPU_ID_M1 = "cuda:6"
    GPU_ID_M2 = "cuda:7"

    # Model 0 - Pre-trained model
    modelM0.to(GPU_ID_M0)

    # Model 1 - Training on
    modelM1.to(GPU_ID_M1)

    # Model 2 - Pre-trained model
    modelM2.to(GPU_ID_M2)

    best_model_wts = ""
    best_acc = 0.0
    best_val_loss = 99999
    since = time.time()
    criterion = DiceLoss()
    print("Before Training")
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in [0, 1]:
            if phase == 0:
                print("Model In Training mode")
                modelM0.eval()  # Set model to evaluate mode
                modelM1.train()  # Set model to training mode
                modelM2.train()  # Set model to training mode
            else:
                print("Model In Validation mode")
                modelM0.eval()  # Set model to evaluate mode
                modelM1.eval()  # Set model to evaluate mode
                modelM2.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            idx = 0
            for batch in tqdm(dataloaders[phase]):

                # Get Data
                mri_batch, labels_batch, ct = batch
                optimizer.zero_grad()

                """
                    # Section 1
                        Input to model M1     : MRI
                        Input to model M1     : CT images
                        Output from model M1  : Warp Field

                    # Section 2
                        Use warp field on GT MRI -> Pseudo GT
                        Use warp field on CT     -> Pseudo CTMR
    
                        Input to model M2    : CT
                        Input to model M2    : MR or Pseudo CTMR
                        Output from model M2 : Image for training 
    
                        Input to model M0 : Image for training from Model M2
                        Input to model M0 : Pseudo GT
                """
                # Section 1
                input = torch.cat((mri_batch, ct), 0)
                with torch.set_grad_enabled(phase == 0):
                    with autocast(enabled=False):
                        output_warp = modelM1(input.unsqueeze(0).to(GPU_ID_M1))[0]

                    # Section 2
                    labels_batch_input_grid = torch.cat((labels_batch, labels_batch, labels_batch)).unsqueeze(
                        0).permute(0, 2, 3, 4, 1)

                    # warped_MRI = F.grid_sample(mri_batch, output_warp, mode="trilinear")     # ct_actual
                    pseudo_lbl = F.grid_sample(output_warp, labels_batch_input_grid.to(GPU_ID_M1),
                                               mode="bilinear")  # labels_batch
                    # input = torch.cat((warped_MRI, ct_actual), 0) #Attempt 2
                    input = torch.cat((mri_batch, ct), 0)  # Attempt 1
                    with autocast(enabled=True):
                        output_mergeCTMR = modelM2(input.unsqueeze(0).to(GPU_ID_M2))[0]
                        output_ct = modelM0(output_mergeCTMR.unsqueeze(1).to(GPU_ID_M0))[0].squeeze()

                    loss, acc = criterion(output_ct, pseudo_lbl.squeeze().to(GPU_ID_M0))

                    # backward + optimize only if in training phase
                    if phase == 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        if epoch % 1 == 0 and log:
                            mri = mri_batch.squeeze()[8:9, :, :]
                            ct = ct.squeeze()[8:9, :, :]
                            mri_op = output_ct[8:9, :, :].float()
                            ct_op = output_ct[0].squeeze()[8:9, :, :].detach().cpu().float()
                            mri_lbl = labels_batch.squeeze()[8:9, :, :].detach().cpu()
                            op = pseudo_lbl[8:9, :, :]
                            mri_op = (mri_op - mri_op.min()) / (mri_op.max() - mri_op.min())
                            ct_op = (ct_op - ct_op.min()) / (ct_op.max() - ct_op.min())
                            fig = saveImage(mri, mri_op, mri_lbl, ct, ct_op, op)
                            text = "Images on - " + str(epoch)
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

            print('{} Loss: {:.4f}'.format(mode, epoch_loss))

            # deep copy the model
            if phase == 1 and (epoch_acc > best_acc or epoch_loss < best_val_loss):
                print("Saving the best model weights")
                best_val_loss = epoch_loss
                best_acc = epoch_acc
                torch.save(modelM1.state_dict(), M1_bw_path)
                torch.save(modelM2.state_dict(), M2_bw_path)

        if epoch % 10 == 0:
            print("Saving the model")
            # save the model
            torch.save(modelM1.state_dict(), M1_model_path)
            torch.save(modelM2.state_dict(), M2_model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # save the model
    print("Saving the model before exiting")
    torch.save(modelM1.state_dict(), M1_model_path)
    torch.save(modelM2.state_dict(), M2_model_path)
