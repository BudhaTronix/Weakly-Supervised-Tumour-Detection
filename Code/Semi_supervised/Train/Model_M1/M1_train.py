import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)
from skimage.filters import threshold_otsu
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Code.Utils.loss import DiceLoss, focal_tversky_loss

scaler = GradScaler()


def saveImage(mri, mri_lbl, ct, ctmri_merge, ct_op, pseudo_gt, ct_gt=None, isChaos=False):
    # create grid of images
    figure = plt.figure(figsize=(10, 10))

    plt.subplot(331, title="Inp : MRI")
    plt.grid(False)
    plt.imshow(mri.permute(1, 2, 0), cmap="gray")

    plt.subplot(332, title="Inp : CT")
    plt.grid(False)
    plt.imshow(ct.permute(1, 2, 0), cmap="gray")

    plt.subplot(333, title="CT-MR")
    plt.grid(False)
    plt.imshow(ctmri_merge.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(334, title="MRI LBL")
    plt.grid(False)
    plt.imshow(mri_lbl.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(336, title="Pseudo CT LBL")
    plt.grid(False)
    plt.imshow(pseudo_gt.permute(1, 2, 0), cmap="gray")

    plt.subplot(339, title="Output")
    plt.grid(False)
    plt.imshow(ct_op.permute(1, 2, 0).to(torch.float), cmap="gray")

    if isChaos:
        plt.subplot(335, title="CT LBL")
        plt.grid(False)
        plt.imshow(ct_gt.permute(1, 2, 0).to(torch.float), cmap="gray")

    return figure


def saveModel(modelM1, path):
    torch.save({"feature_extractor_training": modelM1.feature_extractor_training.state_dict(),
                "scg_training": modelM1.scg_training.state_dict(),
                "upsampler1_training": modelM1.upsampler1_training.state_dict(),
                "upsampler2_training": modelM1.upsampler2_training.state_dict(),
                "upsampler3_training": modelM1.upsampler3_training.state_dict(),
                "upsampler4_training": modelM1.upsampler4_training.state_dict(),
                "upsampler5_training": modelM1.upsampler5_training.state_dict(),
                "graph_layers1_training": modelM1.graph_layers1_training.state_dict(),
                "graph_layers2_training": modelM1.graph_layers2_training.state_dict(),
                "conv_decoder1_training": modelM1.conv_decoder1_training.state_dict(),
                "conv_decoder2_training": modelM1.conv_decoder2_training.state_dict(),
                "conv_decoder3_training": modelM1.conv_decoder3_training.state_dict(),
                "conv_decoder4_training": modelM1.conv_decoder4_training.state_dict(),
                "conv_decoder5_training": modelM1.conv_decoder5_training.state_dict(),
                "conv_decoder6_training": modelM1.conv_decoder6_training.state_dict()}, path)


def train(dataloaders, M1_model_path, M1_bw_path, num_epochs, modelM0, modelM1, optimizer, isChaos,
          isM0Frozen, isM1Frozen, GPU_ID, loss_fn="Dice", model_type="DeepSup", log=False, logPath="",
          M0_model_path=None, M0_bw_path=None):
    if log:
        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        TBLOGDIR = logPath + "{}".format(start_time)
        writer = SummaryWriter(TBLOGDIR)
        logging.info("Tensorboard path : " + str(TBLOGDIR))
    best_acc = 0.0
    best_val_loss_0 = 99999
    best_val_loss_1 = 99999
    since = time.time()
    if loss_fn == "TFL":
        criterion = focal_tversky_loss
    else:
        criterion = DiceLoss()
    getDice = DiceLoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in [0, 1]:
            if phase == 0:
                if isM0Frozen and isM1Frozen:
                    modelM0.train()
                    # modelM1.train()
                if isM0Frozen and not isM1Frozen:
                    modelM0.eval()
                    # modelM1.train()
                if isM1Frozen and not isM0Frozen:
                    modelM0.train()
                    # modelM1.eval()
            else:
                modelM0.eval()
                # modelM1.eval()
            running_loss_0 = 0.0
            running_loss_1 = 0.0
            running_corrects = 0
            # Iterate over data.
            idx = 0
            for batch in tqdm(dataloaders[phase]):
                # Get Data
                if isChaos:
                    mri_batch, labels_batch, ct_batch, ct_gt_batch = batch
                else:
                    mri_batch, labels_batch, ct_batch, ct_gt_batch = batch

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 0):
                    with autocast(enabled=False):
                        if len(labels_batch.shape) < 5:
                            labels_batch = labels_batch.unsqueeze(0)
                        loss_1, fully_warped_image_yx, pseudo_lbl = modelM1.lossCal(ct_batch, mri_batch, labels_batch)
                        fully_warped_image_yx = (fully_warped_image_yx - fully_warped_image_yx.min()) / \
                                                (fully_warped_image_yx.max() - fully_warped_image_yx.min())

                        output_ct = modelM0(fully_warped_image_yx.to(GPU_ID))

                        if model_type == "DeepSup":
                            loss_0 = (criterion(output_ct[0], pseudo_lbl[:, :, ::8, ::8, ::8])
                                      + criterion(output_ct[1], pseudo_lbl[:, :, ::4, ::4, ::4])
                                      + criterion(output_ct[2], pseudo_lbl[:, :, ::2, ::2, ::2])
                                      + criterion(output_ct[3], pseudo_lbl)) / 4.
                        else:
                            loss_0 = criterion(output_ct.squeeze(), pseudo_lbl.squeeze().to(GPU_ID))

                        if not isM0Frozen and not isM1Frozen:
                            total_loss = loss_0 + loss_1  # (loss_0 * 0.2) + (loss_1 * 0.8)
                        if isM0Frozen and not isM1Frozen:
                            total_loss = loss_1
                            loss_0 = loss_0.detach()
                        if isM1Frozen and not isM0Frozen:
                            total_loss = loss_0
                            loss_1 = loss_1.detach()

                        if isChaos:
                            acc_gt = 1 - getDice(ct_gt_batch.squeeze().to(GPU_ID), pseudo_lbl.squeeze().to(GPU_ID))

                    if phase == 0:
                        if autocast:
                            scaler.scale(total_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            total_loss.backward()
                            optimizer.step()

                    if epoch % 10 == 0 and log and idx == 0:
                        temp = labels_batch.squeeze().detach().cpu()
                        slice = 0
                        for i in range(len(temp)):
                            if temp[i].max() == 1:
                                slice = i
                                break
                        # Thresholding OTSU
                        thresh = threshold_otsu(output_ct.cpu().detach().numpy())
                        output_ct = torch.Tensor(output_ct.cpu().detach().numpy() > thresh).to(GPU_ID)
                        mri = mri_batch.squeeze()[slice, :, :].unsqueeze(0)
                        ct = ct_batch.squeeze()[slice, :, :].unsqueeze(0)
                        ctmri_merge = fully_warped_image_yx.squeeze()[slice, :, :].unsqueeze(
                            0).float().clone().detach().cpu()
                        if model_type == "DeepSup":
                            ct_op = output_ct[3].squeeze()[slice, :, :].unsqueeze(0).clone().detach().cpu().float()
                        else:
                            ct_op = output_ct.squeeze()[slice, :, :].unsqueeze(0).clone().detach().cpu().float()
                        mri_lbl = labels_batch.squeeze()[slice, :, :].unsqueeze(0).clone().detach().cpu()
                        pseudo_gt = pseudo_lbl.squeeze()[slice, :, :].unsqueeze(0).clone().detach().cpu()

                        ctmri_merge = (ctmri_merge - ctmri_merge.min()) / (ctmri_merge.max() - ctmri_merge.min())
                        ct_op = (ct_op - ct_op.min()) / (ct_op.max() - ct_op.min())

                        # if isChaos:
                        ct_gt = ct_gt_batch.squeeze()[slice, :, :].unsqueeze(0)
                        fig = saveImage(mri, mri_lbl, ct, ctmri_merge, ct_op, pseudo_gt, ct_gt, isChaos=True)
                        #else:
                        #fig = saveImage(mri, mri_lbl, ct, ctmri_merge, ct_op, pseudo_gt)
                        text = "Images on - " + str(epoch) + " Phase : " + str(phase)
                        writer.add_figure(text, fig, epoch)

                    # statistics
                    running_loss_0 += loss_0.item()
                    running_loss_1 += loss_1.item()

                    if isChaos:
                        running_corrects += acc_gt
                    idx += 1

            epoch_loss_0 = running_loss_0 / len(dataloaders[phase])
            epoch_loss_1 = running_loss_1 / len(dataloaders[phase])
            if isChaos:
                epoch_acc_gt = running_corrects / len(dataloaders[phase])
            if phase == 0:
                mode = "Train"
                if log:
                    writer.add_scalar("Train/Loss_0", epoch_loss_0, epoch)
                    writer.add_scalar("Train/Loss_1", epoch_loss_1, epoch)
                    if isChaos:
                        writer.add_scalar("Train/Acc_GT", epoch_acc_gt, epoch)
            else:
                mode = "Val"
                if log:
                    writer.add_scalar("Validation/Loss_0", epoch_loss_0, epoch)
                    writer.add_scalar("Validation/Loss_1", epoch_loss_1, epoch)
                    if isChaos:
                        writer.add_scalar("Validation/Acc_GT", epoch_acc_gt, epoch)

            logging.info(
                'Epoch: {} Mode: {} Loss_0: {:.4f} Loss_1: {:.4f}'.format(epoch, mode, epoch_loss_0, epoch_loss_1))

            # deep copy the model
            if phase == 1:
                if epoch_loss_0 < best_val_loss_0 and not isM0Frozen:
                    logging.info("Saving the best model weights of Model 0")
                    best_val_loss_0 = epoch_loss_0
                    torch.save(modelM0.state_dict(), M0_bw_path)
                if epoch_loss_1 < best_val_loss_1 and not isM1Frozen:
                    logging.info("Saving the best model weights of Model 1")
                    best_val_loss_1 = epoch_loss_1
                    saveModel(modelM1, M1_bw_path)

        if epoch % 10 == 0:
            logging.info("Saving the model")
            # save the models
            if not isM0Frozen:
                torch.save(modelM0.state_dict(), M0_model_path)
            if not isM1Frozen:
                saveModel(modelM1, M1_model_path)

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # save the model
    logging.info("Saving the models before exiting")
    if not isM0Frozen:
        torch.save(modelM0.state_dict(), M0_model_path)
    if not isM1Frozen:
        saveModel(modelM1, M1_model_path)

    logging.info("############################## END Model Training ##############################")
