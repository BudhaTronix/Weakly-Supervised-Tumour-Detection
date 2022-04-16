import os
import sys
import time
import random
import numpy
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
print(ROOT_DIR)
sys.path.insert(1, ROOT_DIR + "/")
from Code.Utils.loss import DiceLoss
from Code.Semi_supervised.mscgunet.train import fullmodel_one_epoch_run

scaler = GradScaler()

torch.cuda.manual_seed(42)
torch.manual_seed(42)
numpy.random.seed(seed=42)
random.seed(42)


# Change image plot - mri with label
# Give meaningful names to variables
def saveImage(mri, mri_lbl, ct, ctmri_merge, ct_op, pseudo_gt):
    # create grid of images
    figure = plt.figure(figsize=(10, 10))

    plt.subplot(231, title="MRI")
    plt.grid(False)
    plt.imshow(mri.permute(1, 2, 0), cmap="gray")

    plt.subplot(232, title="CT MR Merg")
    plt.grid(False)
    plt.imshow(ctmri_merge.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(233, title="MRI LBL")
    plt.grid(False)
    plt.imshow(mri_lbl.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(234, title="CT")
    plt.grid(False)
    plt.imshow(ct.permute(1, 2, 0), cmap="gray")

    plt.subplot(235, title="CT OP")
    plt.grid(False)
    plt.imshow(ct_op.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(236, title="Pseudo CT LBL")
    plt.grid(False)
    plt.imshow(torch.tensor(pseudo_gt).permute(1, 2, 0), cmap="gray")

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

    # Model 1 -
    # modelM1.to(GPU_ID_M1)

    # Model 2 -
    # modelM2.to(GPU_ID_M2)

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
                modelM0.train()  # Set model to evaluate mode
                # modelM1.train()  # Set model to training mode
                # modelM2.train()  # Set model to training mode
            else:
                print("Model In Validation mode")
                modelM0.eval()  # Set model to evaluate mode
                # modelM1.eval()  # Set model to evaluate mode
                # modelM2.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            idx = 0
            for batch in tqdm(dataloaders[phase]):

                # Get Data
                mri_batch, labels_batch, ct = batch
                optimizer.zero_grad()

                """
                function - MRI , CT
                
                output - loss, fullywarped, flow
                """

                # Section 1
                # input = torch.cat((ct, mri_batch), 1)
                with torch.set_grad_enabled(phase == 0):
                    with autocast(enabled=False):
                        # output_warp = modelM1(input.to(GPU_ID_M1))
                        total_loss, fully_warped_image_yx, psuedo_lbl = fullmodel_one_epoch_run(ct, mri_batch,
                                                                                                labels_batch)

                        # Section 2
                        # grid should be: N, D_\text{out}, H_\text{out}, W_\text{out}, 3, but your model is giving you N, C, D, H, W where C=3
                        # output_warp = output_warp.permute(0, 2, 3, 4, 1)
                        # warped_MRI = F.grid_sample(mri_batch.to(GPU_ID_M1), output_warp, mode="bilinear")  # ct_actual
                        # pseudo_lbl = F.grid_sample(labels_batch.to(GPU_ID_M1), output_warp, mode="bilinear")  # labels_batch

                        # input = torch.cat((ct.to(GPU_ID_M1), warped_MRI), 1)  # Attempt 1
                        # input = torch.cat((ct, mri_batch), 1)  # Attempt 2
                        # with autocast(enabled=False): q2
                        # output_mergeCTMR = modelM2(fully_warped_image_yx.to(GPU_ID_M2))
                        output_ct = modelM0(fully_warped_image_yx.to(GPU_ID_M0)).squeeze()

                        loss_1, acc = criterion(output_ct, psuedo_lbl.squeeze().to(GPU_ID_M0))
                        # _, acc = criterion(output_ct, pseudo_lbl.squeeze().to(GPU_ID_M0)) # Compare between actual CT label with output_ct

                        loss = loss_1  # + total_loss

                    # backward + optimize only if in training phase
                    if phase == 0:
                        if autocast:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        if epoch % 1 == 0 and log and idx == 0:
                            temp = labels_batch.squeeze().detach().cpu()
                            slice = 0
                            for i in range(len(temp)):
                                if temp[i].max() == 1:
                                    slice = i
                                    break
                            mri = mri_batch.squeeze()[slice, :, :].unsqueeze(0)
                            ct = ct.squeeze()[slice, :, :].unsqueeze(0)
                            ctmri_merge = fully_warped_image_yx.squeeze()[slice, :, :].unsqueeze(
                                0).float().detach().cpu()
                            ct_op = output_ct[slice, :, :].unsqueeze(0).detach().cpu().float()
                            mri_lbl = labels_batch.squeeze()[slice, :, :].unsqueeze(0).detach().cpu()
                            pseudo_gt = psuedo_lbl.squeeze()[slice, :, :].unsqueeze(0).detach().cpu()
                            ctmri_merge = (ctmri_merge - ctmri_merge.min()) / (ctmri_merge.max() - ctmri_merge.min())
                            ct_op = (ct_op - ct_op.min()) / (ct_op.max() - ct_op.min())
                            fig = saveImage(mri, mri_lbl, ct, ctmri_merge, ct_op, pseudo_gt)
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
