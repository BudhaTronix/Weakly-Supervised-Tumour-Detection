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

from Code.Utils.loss import DiceLoss, focal_tversky_loss

scaler = GradScaler()

torch.cuda.manual_seed(42)
torch.manual_seed(42)
numpy.random.seed(seed=42)
random.seed(42)


def saveImage(ct, ct_op, ct_gt):
    # create grid of images
    figure = plt.figure(figsize=(10, 10))

    plt.subplot(311, title="Inp : CT")
    plt.grid(False)
    plt.imshow(ct.permute(1, 2, 0), cmap="gray")

    plt.subplot(312, title="CT LBL")
    plt.grid(False)
    plt.imshow(ct_gt.permute(1, 2, 0).to(torch.float), cmap="gray")

    plt.subplot(313, title="Output")
    plt.grid(False)
    plt.imshow(ct_op.permute(1, 2, 0).to(torch.float), cmap="gray")

    return figure


def test(dataloaders, modelM0, model_type, log=False, logPath="", device="cuda"):
    if log:
        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        TBLOGDIR = logPath + "{}".format(start_time)
        writer = SummaryWriter(TBLOGDIR)
        logging.info("Tensorboard for testing path : ", TBLOGDIR)

    torch.cuda.empty_cache()
    GPU_ID_M0 = device
    modelM0.to(device)
    since = time.time()
    criterion = focal_tversky_loss
    getDice = DiceLoss()
    jaccard = JaccardIndex(num_classes=2)

    idx = 0
    running_loss_0 = 0
    running_corrects = 0
    for batch in dataloaders:
        # Get Data
        ct_batch, ct_gt_batch = batch
        with torch.no_grad():
            with autocast(enabled=False):
                output_ct = modelM0(ct_batch.unsqueeze(0).to(GPU_ID_M0))
                if model_type == "DeepSup":
                    loss_0 = (criterion(output_ct[0].squeeze(0), ct_gt_batch[ :, ::8, ::8, ::8].to(GPU_ID_M0))
                              + criterion(output_ct[1].squeeze(0), ct_gt_batch[ :, ::4, ::4, ::4].to(GPU_ID_M0))
                              + criterion(output_ct[2].squeeze(0), ct_gt_batch[:, ::2, ::2, ::2].to(GPU_ID_M0))
                              + criterion(output_ct[3].squeeze(0), ct_gt_batch.to(GPU_ID_M0))) / 4.
                    # Dice Score
                    acc_gt = 1 - getDice(output_ct[3].squeeze().to(GPU_ID_M0), ct_gt_batch.squeeze().to(GPU_ID_M0))
                    # Jaccard Index
                    j_value = jaccard(output_ct[3].cpu().squeeze(), ct_gt_batch.squeeze().cpu().type(torch.int))
                else:
                    loss_0 = criterion(output_ct.squeeze(), ct_gt_batch.squeeze().to(GPU_ID_M0))
                    # Dice Score
                    acc_gt = 1 - getDice(output_ct.squeeze().to(GPU_ID_M0), ct_gt_batch.squeeze().to(GPU_ID_M0))
                    # Jaccard Index
                    j_value = jaccard(output_ct.cpu().squeeze(), ct_gt_batch.squeeze().cpu().type(torch.int))

                print("File: ", idx, "  Dice: ", acc_gt.item(), "  Jaccard: ", j_value.item(), "  Focal_Tr: ",
                      loss_0.item())
                logging.debug("File: " + str(idx) + "  Dice: " + str(acc_gt.item()) + "  Jaccard: " + str(
                    j_value.item()) + "  Focal_Tr: " + str(loss_0.item()))

                torch.cuda.empty_cache()

            if log:
                temp = ct_gt_batch.squeeze().detach().cpu()
                slice = 0
                for i in range(len(temp)):
                    if temp[i].max() == 1:
                        slice = i
                        break
                ct = ct_batch.squeeze()[slice, :, :].unsqueeze(0)
                if model_type == "DeepSup":
                    ct_op = output_ct[3].squeeze()[slice, :, :].unsqueeze(0).clone().detach().cpu().float()
                else:
                    ct_op = output_ct.squeeze()[slice, :, :].unsqueeze(0).clone().detach().cpu().float()
                ct_gt = ct_gt_batch.squeeze()[slice, :, :].unsqueeze(0)

                ct_op = (ct_op - ct_op.min()) / (ct_op.max() - ct_op.min())

                fig = saveImage(ct, ct_op, ct_gt)
                text = "Images : " + str(idx)
                writer.add_figure(text, fig, idx)

            if log:
                writer.add_scalar("Loss_0", loss_0.item(), idx)
                writer.add_scalar("Acc_GT", acc_gt.item(), idx)
                writer.add_scalar("Jaccard", j_value, idx)

            # statistics
            running_loss_0 += loss_0.item()
            running_corrects += acc_gt.item()
            idx += 1

    print("Overall loss 0: ", running_loss_0 / len(dataloaders))
    print("Overall Accuracy : ", running_corrects / len(dataloaders))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    logging.debug("Overall loss 0   : " + str(running_loss_0 / len(dataloaders)))
    logging.debug("Overall Accuracy : " + str(running_corrects / len(dataloaders)))
    logging.debug('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info("############################# END Model Testing #############################")
