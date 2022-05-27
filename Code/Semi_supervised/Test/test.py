import time
import random
import numpy
from datetime import datetime
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch
import logging

torch.set_num_threads(1)

from Code.Utils.loss import DiceLoss

scaler = GradScaler()

torch.cuda.manual_seed(42)
torch.manual_seed(42)
numpy.random.seed(seed=42)
random.seed(42)


def saveImage(mri, mri_lbl, ct, ctmri_merge, ct_op, pseudo_gt, ct_gt):
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

    plt.subplot(335, title="CT LBL")
    plt.grid(False)
    plt.imshow(ct_gt.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(336, title="Pseudo CT LBL")
    plt.grid(False)
    plt.imshow(pseudo_gt.permute(1, 2, 0), cmap="gray")

    plt.subplot(339, title="Output")
    plt.grid(False)
    plt.imshow(ct_op.permute(1, 2, 0).to(torch.float), cmap="gray")

    return figure


def test(dataloaders, modelM0, modelM1, log=False, logPath=""):
    if log:
        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        TBLOGDIR = logPath + "{}".format(start_time)
        writer = SummaryWriter(TBLOGDIR)
        logging.info("Tensorboard for testing path : ", TBLOGDIR)

    GPU_ID_M0 = "cuda:" + str(next(modelM0.parameters()).device.index)
    since = time.time()
    criterion = DiceLoss()
    jaccard = JaccardIndex(num_classes=2)

    idx = 0
    running_loss_0 = 0
    running_loss_1 = 0
    running_corrects = 0
    ids = []
    for batch in dataloaders:
        # Get Data
        mri_batch, labels_batch, ct_batch, ct_gt_batch, id = batch
        ids.append(id.item())

        with autocast(enabled=False):
            loss_1, fully_warped_image_yx, pseudo_lbl = modelM1.lossCal(ct_batch, mri_batch, labels_batch)
            output_ct = modelM0(fully_warped_image_yx.to(GPU_ID_M0)).squeeze()
            loss_0, _ = criterion(output_ct, pseudo_lbl.squeeze().to(GPU_ID_M0))

            # Dice Score
            _, acc_gt = criterion(ct_gt_batch.squeeze().to(GPU_ID_M0), pseudo_lbl.squeeze().to(GPU_ID_M0))
            # Jaccard Index
            j_value = jaccard(ct_gt_batch.squeeze(), pseudo_lbl.squeeze().cpu().type(torch.int))
            print("File: ", idx, "  Dice: ", acc_gt.item(), "  Jaccard: ", j_value.item())
            logging.debug("File: " + str(idx) + "  Dice: " + str(acc_gt.item()) + "  Jaccard: " + str(j_value.item()))

        if log:
            temp = labels_batch.squeeze().detach().cpu()
            slice = 0
            for i in range(len(temp)):
                if temp[i].max() == 1:
                    slice = i
                    break
            mri = mri_batch.squeeze()[slice, :, :].unsqueeze(0)
            ct = ct_batch.squeeze()[slice, :, :].unsqueeze(0)
            ctmri_merge = fully_warped_image_yx.squeeze()[slice, :, :].unsqueeze(0).float().clone().detach().cpu()
            ct_op = output_ct[slice, :, :].unsqueeze(0).clone().detach().cpu().float()
            mri_lbl = labels_batch.squeeze()[slice, :, :].unsqueeze(0).clone().detach().cpu()
            pseudo_gt = pseudo_lbl.squeeze()[slice, :, :].unsqueeze(0).clone().detach().cpu()
            ct_gt = ct_gt_batch.squeeze()[slice, :, :].unsqueeze(0)

            ctmri_merge = (ctmri_merge - ctmri_merge.min()) / (ctmri_merge.max() - ctmri_merge.min())
            ct_op = (ct_op - ct_op.min()) / (ct_op.max() - ct_op.min())

            fig = saveImage(mri, mri_lbl, ct, ctmri_merge, ct_op, pseudo_gt, ct_gt)
            text = "Images : " + str(idx)
            writer.add_figure(text, fig, idx)

        if log:
            writer.add_scalar("Loss_0", loss_0.item(), idx)
            writer.add_scalar("Loss_1", loss_1.item(), idx)
            writer.add_scalar("Acc_GT", acc_gt.item(), idx)
            writer.add_scalar("Jaccard", j_value, idx)

        # statistics
        running_loss_0 += loss_0.item()
        running_loss_1 += loss_1.item()
        running_corrects += acc_gt.item()
        idx += 1

    print("Overall loss 0: ", running_loss_0 / len(dataloaders))
    print("Overall loss 1: ", running_loss_1 / len(dataloaders))
    print("Overall Accuracy : ", running_corrects / len(dataloaders))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("IDs used in testing : ", ids)

    logging.debug("Overall loss 0   : " + str(running_loss_0 / len(dataloaders)))
    logging.debug("Overall loss 1   : " + str(running_loss_1 / len(dataloaders)))
    logging.debug("Overall Accuracy : " + str(running_corrects / len(dataloaders)))
    logging.debug('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.debug("IDs used in testing : " + str(ids))
    logging.info("############################# END Model Testing #############################")
