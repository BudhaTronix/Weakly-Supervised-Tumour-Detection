import torch
from torch import nn
import torch.nn.functional as F


class UnetConv3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm, is_leaky=False):
        super(UnetConv3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels) if is_batchnorm else nn.Sequential()
        self.act1 = nn.PReLU(out_channels) if is_leaky else nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels) if is_batchnorm else nn.Sequential()
        self.act2 = nn.PReLU(out_channels) if is_leaky else nn.ReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        return self.act2(self.bn2(self.conv2(x)))


class UnetGatingSignal(torch.nn.Module):
    def __init__(self, in_channels, is_batchnorm, is_leaky=False):
        super(UnetGatingSignal, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, 1)
        self.bn = nn.BatchNorm3d(in_channels) if is_batchnorm else nn.Sequential()
        self.act = nn.PReLU(in_channels) if is_leaky else nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AttnGatingBlock(torch.nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels, is_leaky=False):
        super(AttnGatingBlock, self).__init__()
        self.conv1 = nn.Conv3d(x_channels, inter_channels, 2, 2)
        self.conv2 = nn.Conv3d(g_channels, inter_channels, 1)
        self.conv3 = nn.Conv3d(inter_channels, 1, 1)
        self.conv4 = nn.Conv3d(x_channels, x_channels, 1)
        self.bn1 = nn.BatchNorm3d(x_channels)
        self.act_xg = nn.PReLU(inter_channels) if is_leaky else nn.ReLU()

    def forward(self, x, g):
        theta_x = self.conv1(x)
        phi_g = self.conv2(g)

        concat_xg = theta_x + phi_g
        act_xg = self.act_xg(concat_xg)
        psi = self.conv3(act_xg)
        sigmoid_xg = torch.sigmoid(psi)

        upsample_psi = F.interpolate(sigmoid_xg, scale_factor=2, mode='nearest')  #F.interpolate(sigmoid_xg, scale_factor=2, mode='trilinear')
        upsample_psi = upsample_psi.repeat(1, x.shape[1], 1, 1, 1)

        y = torch.mul(upsample_psi, x)
        result = self.conv4(y)
        result_bn = self.bn1(result)
        return result_bn


class DeepSupAttentionUnet(torch.nn.Module):
    ''' Implementation of http://arxiv.org/abs/1810.07842
    '''

    def __init__(self, in_channels, out_channels, is_batchnorm=True, is_leaky=True, finalact="sigmoid"):
        super(DeepSupAttentionUnet, self).__init__()
        self.conv1 = UnetConv3D(in_channels, 32, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.conv_scale2 = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.conv2 = UnetConv3D(64 + 32, 64, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.conv_scale3 = nn.Conv3d(in_channels, 128, 3, padding=1)
        self.conv3 = UnetConv3D(128 + 64, 128, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.conv_scale4 = nn.Conv3d(in_channels, 256, 3, padding=1)
        self.conv4 = UnetConv3D(256 + 128, 256, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.center = UnetConv3D(256, 512, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.g1 = UnetGatingSignal(512, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.attn1 = AttnGatingBlock(256, 512, 128, is_leaky=is_leaky)
        self.up1 = nn.ConvTranspose3d(512, 32, 2, 2)
        self.g2 = UnetGatingSignal(256 + 32, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.attn2 = AttnGatingBlock(128, 256 + 32, 64, is_leaky=is_leaky)
        self.up2 = nn.ConvTranspose3d(256 + 32, 64, 2, 2)
        self.g3 = UnetGatingSignal(128 + 64, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.attn3 = AttnGatingBlock(64, 128 + 64, 32, is_leaky=is_leaky)
        self.up3 = nn.ConvTranspose3d(128 + 64, 32, 2, 2)
        self.up4 = nn.ConvTranspose3d(64 + 32, 32, 2, 2)
        self.conv6 = UnetConv3D(288, 256, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.conv7 = UnetConv3D(192, 128, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.conv8 = UnetConv3D(96, 64, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.conv9 = UnetConv3D(64, 32, is_batchnorm=is_batchnorm, is_leaky=is_leaky)
        self.pred1 = nn.Conv3d(256, out_channels, 1)
        self.pred2 = nn.Conv3d(128, out_channels, 1)
        self.pred3 = nn.Conv3d(64, out_channels, 1)
        self.final = nn.Conv3d(32, out_channels, 1)

        self.act1 = nn.PReLU(64) if is_leaky else nn.ReLU()
        self.act2 = nn.PReLU(128) if is_leaky else nn.ReLU()
        self.act3 = nn.PReLU(256) if is_leaky else nn.ReLU()
        self.act4 = nn.PReLU(32) if is_leaky else nn.ReLU()
        self.act5 = nn.PReLU(64) if is_leaky else nn.ReLU()
        self.act6 = nn.PReLU(32) if is_leaky else nn.ReLU()
        self.act7 = nn.PReLU(32) if is_leaky else nn.ReLU()

        self.finalact = finalact

    def forward(self, x):
        scale_img_2 = F.avg_pool3d(x, 2)
        scale_img_3 = F.avg_pool3d(scale_img_2, 2)
        scale_img_4 = F.avg_pool3d(scale_img_3, 2)

        conv1 = self.conv1(x)
        pool1 = F.max_pool3d(conv1, 2)

        input2 = self.act1(self.conv_scale2(scale_img_2))
        input2 = torch.cat([input2, pool1], 1)
        conv2 = self.conv2(input2)
        pool2 = F.max_pool3d(conv2, 2)

        input3 = self.act2(self.conv_scale3(scale_img_3))
        input3 = torch.cat([input3, pool2], 1)
        conv3 = self.conv3(input3)
        pool3 = F.max_pool3d(conv3, 2)

        input4 = self.act3(self.conv_scale4(scale_img_4))
        input4 = torch.cat([input4, pool3], 1)
        conv4 = self.conv4(input4)
        pool4 = F.max_pool3d(conv4, 2)

        center = self.center(pool4)

        g1 = self.g1(center)
        attn1 = self.attn1(conv4, g1)
        up1 = torch.cat([self.act4(self.up1(center)), attn1], 1)

        g2 = self.g2(up1)
        attn2 = self.attn2(conv3, g2)
        up2 = torch.cat([self.act5(self.up2(up1)), attn2], 1)

        g3 = self.g3(up2)
        attn3 = self.attn3(conv2, g3)
        up3 = torch.cat([self.act6(self.up3(up2)), attn3], 1)

        up4 = torch.cat([self.act7(self.up4(up3)), conv1], 1)

        conv6 = self.conv6(up1)
        conv7 = self.conv7(up2)
        conv8 = self.conv8(up3)
        conv9 = self.conv9(up4)

        out6 = self.pred1(conv6)
        out7 = self.pred2(conv7)
        out8 = self.pred3(conv8)
        out9 = self.final(conv9)

        if self.finalact == "sigmoid":
            return torch.sigmoid(out6), torch.sigmoid(out7), torch.sigmoid(out8), torch.sigmoid(out9)
        elif self.finalact == "softmax":
            return F.softmax(out6, -3), F.softmax(out7, -3), F.softmax(out8, -3), F.softmax(out9, -3)
        else:
            return out6, out7, out8, out9

    @staticmethod
    def forward_pass(model_forward, img, gt, criterion):
        prediction = model_forward(img)

        loss = (criterion(prediction[0], gt[:, :, ::8, ::8, ::8])
                + criterion(prediction[1], gt[:, :, ::4, ::4, ::4])
                + criterion(prediction[2], gt[:, :, ::2, ::2, ::2])
                + criterion(prediction[3], gt)) / 4.

        return loss, prediction[3]


"""if __name__ == '__main__':
    aunet = DeepSupAttentionUnet(1, 1).cuda()
    print(aunet(torch.zeros(1, 1, 64, 64, 64, device='cuda'))[-1].shape)
"""