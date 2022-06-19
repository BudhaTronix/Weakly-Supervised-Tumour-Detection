import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss


def jaccard_loss(pred, target):
    """This definition generalise to real valued pred and target vector.
        This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1

    nclasses = pred.shape[1]
    loss = 0.
    for c in range(nclasses):
        # have to use contiguous since they may from a torch.view op
        iflat = pred[:, c].contiguous().view(-1)
        tflat = target[:, c].contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        jac = (intersection + smooth) / (A_sum + B_sum - intersection + smooth)
        loss += 1 - jac
    return loss


def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=4. / 3.):
    smooth = 1.

    nclasses = pred.shape[1]
    ftl = 0.
    for c in range(nclasses):
        pflat = pred[:, c].contiguous().view(-1)
        gflat = target[:, c].contiguous().view(-1)

        intersection = (pflat * gflat).sum()
        non_p_g = ((1. - pflat) * gflat).sum()
        p_non_g = (pflat * (1. - gflat)).sum()

        ti = (intersection + smooth) / (
                    torch.finfo(torch.float32).eps + intersection + alpha * non_p_g + beta * p_non_g + smooth)
        ftl += (1. - ti) ** (1. / gamma + torch.finfo(torch.float32).eps)
    return ftl
