from model import *
from losses import *
from layers import *
import glob
import os
import torch
import time
import wandb
import argparse


def fullmodel_one_epoch_run(CT, MRI, MRI_LBL, epoch=1):
    X = CT
    Y = MRI

    X = X.float().to("cuda")
    Y = Y.float().to("cuda")
    Ylbl = MRI_LBL.float().to("cuda")

    half_shape = tuple([i // 2 for i in X.shape[2:]])
    X_64 = F.interpolate(X, half_shape, mode='trilinear', align_corners=False)
    Y_64 = F.interpolate(Y, half_shape, mode='trilinear', align_corners=False)

    # ============================================ X-Y START =================================================================

    enc_1_xy, enc_2_xy, enc_3_xy, enc_4_xy, enc_5_xy, enc_6_xy = feature_extractor_training(X, Y)
    A_xy, gx_xy, scg_loss_xy, z_hat_xy = scg_training(enc_6_xy)
    B_xy, C_xy, H_xy, W_xy, D_xy = enc_6_xy.size()
    gop_layers1_xy, A_layers1_xy = graph_layers1_training((gx_xy.reshape(B_xy, -1, C_xy), A_xy))
    gop_layers2_xy, A_layers2_xy = graph_layers2_training((gop_layers1_xy, A_layers1_xy))

    gop_layers2_xy = torch.bmm(A_layers2_xy, gop_layers2_xy)
    gop_layers2_xy = gop_layers2_xy + z_hat_xy

    # Upward trajectory
    gx_xy = gop_layers2_xy.reshape(B_xy, 9, 4, 4, 4)
    gx_xy = F.interpolate(gx_xy, (H_xy, W_xy, D_xy), mode='trilinear', align_corners=False)

    # Adding information from feature extractor directly to latent space info, this could provide a path for gradients to move faster
    gx_xy = upsampler1_training(gx_xy, enc_5_xy)  # 8
    gx_xy = upsampler2_training(gx_xy, enc_4_xy)  # 16
    gx_xy = upsampler3_training(gx_xy, enc_3_xy)  # 32
    gx_64_xy = upsampler4_training(gx_xy, enc_2_xy)  # 64
    gx_xy = upsampler5_training(gx_64_xy, enc_1_xy)  # 128, 32

    # Concat fixed image to final field before smoothening
    gx_xy = torch.cat((X, gx_xy), 1)

    # Conv decoder last three layers from voxel morph
    gx_xy = conv_decoder1_training(gx_xy)  # 128, 33 --> 16
    gx_xy = conv_decoder2_training(gx_xy)  # 128,16
    dvf_xy = conv_decoder3_training(gx_xy)  # 128,3

    # Conv decoder last three layers from voxel morph 64
    gx_64_xy = conv_decoder4_training(gx_64_xy)  # 128, 33 --> 16
    gx_64_xy = conv_decoder5_training(gx_64_xy)  # 128,16
    dvf_64_xy = conv_decoder6_training(gx_64_xy)  # 128,3

    # vector integration for diffeomorphic field
    pos_flow_xy = resize(dvf_xy)
    integrated_pos_flow_xy = integrate(pos_flow_xy)
    full_flow_xy = fullsize(integrated_pos_flow_xy)

    # vector integration  64
    pos_flow_64_xy = resize(dvf_64_xy)
    integrated_pos_flow_64_xy = integrate_64(pos_flow_64_xy)
    full_flow_64_xy = fullsize(integrated_pos_flow_64_xy)

    # warp
    fully_warped_image_xy = stn_deformable(Y, full_flow_xy)
    fully_warped_image_64_xy = stn_deformable_64(Y_64, full_flow_64_xy)

    # ============================================= X-Y END ====================================================================

    # ============================================= Y-X START ==================================================================

    enc_1_yx, enc_2_yx, enc_3_yx, enc_4_yx, enc_5_yx, enc_6_yx = feature_extractor_training(Y, X)
    A_yx, gx_yx, scg_loss_yx, z_hat_yx = scg_training(enc_6_yx)
    B_yx, C_yx, H_yx, W_yx, D_yx = enc_6_yx.size()
    gop_layers1_yx, A_layers1_yx = graph_layers1_training((gx_yx.reshape(B_yx, -1, C_yx), A_yx))
    gop_layers2_yx, A_layers2_yx = graph_layers2_training((gop_layers1_yx, A_layers1_yx))

    gop_layers2_yx = torch.bmm(A_layers2_yx, gop_layers2_yx)
    gop_layers2_yx = gop_layers2_yx + z_hat_yx

    # Upward trajectory
    gx_yx = gop_layers2_yx.reshape(B_yx, 9, 4, 4, 4)
    gx_yx = F.interpolate(gx_yx, (H_yx, W_yx, D_yx), mode='trilinear', align_corners=False)

    # Adding information from feature extractor directly to latent space info, this could provide a path for gradients to move faster
    gx_yx = upsampler1_training(gx_yx, enc_5_yx)  # 8
    gx_yx = upsampler2_training(gx_yx, enc_4_yx)  # 16
    gx_yx = upsampler3_training(gx_yx, enc_3_yx)  # 32
    gx_64_yx = upsampler4_training(gx_yx, enc_2_yx)  # 64
    gx_yx = upsampler5_training(gx_64_yx, enc_1_yx)  # 128, 32

    # Concat fixed image to final field before smoothening
    gx_yx = torch.cat((Y, gx_yx), 1)

    # Conv decoder last three layers from voxel morph
    gx_yx = conv_decoder1_training(gx_yx)  # 128, 33 --> 16
    gx_yx = conv_decoder2_training(gx_yx)  # 128,16
    dvf_yx = conv_decoder3_training(gx_yx)  # 128,3

    # Conv decoder last three layers from voxel morph 64
    gx_64_yx = conv_decoder4_training(gx_64_yx)  # 128, 33 --> 16
    gx_64_yx = conv_decoder5_training(gx_64_yx)  # 128,16
    dvf_64_yx = conv_decoder6_training(gx_64_yx)  # 128,3

    # vector integration for diffeomorphic field
    pos_flow_yx = resize(dvf_yx)
    integrated_pos_flow_yx = integrate(pos_flow_yx)
    full_flow_yx = fullsize(integrated_pos_flow_yx)

    # vector integration for diffeomorphic field 64
    pos_flow_64_yx = resize(dvf_64_yx)
    integrated_pos_flow_64_yx = integrate_64(pos_flow_64_yx)
    full_flow_64_yx = fullsize(integrated_pos_flow_64_yx)

    #  warp
    fully_warped_image_yx = stn_deformable(X, full_flow_yx)
    fully_warped_image_64_yx = stn_deformable_64(X_64, full_flow_64_yx)

    # ============================================= Y-X END ======================================================================

    cc_loss = similarity_loss(X, fully_warped_image_xy) + similarity_loss(Y, fully_warped_image_yx)
    cc_loss_64 = similarity_loss(X_64, fully_warped_image_64_xy) + similarity_loss(Y_64, fully_warped_image_64_yx)
    sm_loss = smoothness_loss.loss("", full_flow_xy) + smoothness_loss.loss("", full_flow_yx)
    sm_loss_64 = smoothness_loss.loss("", full_flow_64_xy) + smoothness_loss.loss("", full_flow_64_yx)
    scg_loss = scg_loss_xy + scg_loss_yx
    total_loss = hyperparam1 * cc_loss + hyperparam3 * sm_loss + hyperparam2 * scg_loss + cc_loss_64 * -0.5 + sm_loss_64 * 0.5

    ################################################################################################

    psuedo_lbl = stn_deformable(Ylbl, full_flow_xy)
    return total_loss, fully_warped_image_yx, psuedo_lbl


lr = 1e-4
range_flow = 7
hyperparam1 = -1.2
hyperparam2 = 1.0
hyperparam3 = 1.0
hyperparam4 = 10
checkpoint_reload = True

# ========================================= Model Init - START =========================================================

feature_extractor_training = Feature_Extractor(2, 3, 16).to("cuda")

scg_training = SCG_block(in_ch=32, hidden_ch=9, node_size=(4, 4, 4)).to("cuda")

upsampler1_training = convEncoder(9, 32, 32).to("cuda")
upsampler2_training = convEncoder(32, 32, 32).to("cuda")
upsampler3_training = convEncoder(32, 32, 32).to("cuda")
upsampler4_training = convEncoder(32, 32, 32).to("cuda")
upsampler5_training = convEncoder(16, 32, 32).to("cuda")

conv_decoder1_training = convDecoder(33, 16).to("cuda")
conv_decoder2_training = convDecoder(16, 16).to("cuda")
conv_decoder3_training = convDecoder(16, 3).to("cuda")

conv_decoder4_training = convDecoder(32, 16).to("cuda")
conv_decoder5_training = convDecoder(16, 16).to("cuda")
conv_decoder6_training = convDecoder(16, 3).to("cuda")

graph_layers1_training = GCN_Layer(32, 16, bnorm=True, activation=nn.LeakyReLU(0.2), dropout=0.1).to("cuda")
graph_layers2_training = GCN_Layer(16, 9, bnorm=True, activation=nn.LeakyReLU(0.2), dropout=0.1).to("cuda")

stn_deformable = SpatialTransformer(size=(32, 128, 128), is_affine=False).to("cuda")
stn_deformable_64 = SpatialTransformer(size=(16, 64, 64), is_affine=False).to("cuda")

weight_xavier_init(graph_layers1_training, graph_layers2_training, scg_training)

for param in stn_deformable.parameters():
    param.requires_grad = False
    param.volatile = True

# vector integrion to enforce diffeomorphic transform
dim = 3
int_downsize = 2
down_shape = [int(dim / int_downsize) for dim in (32, 128, 128)]
down_shape_64 = [int(dim / int_downsize) for dim in (16, 64, 64)]
resize = ResizeTransform(2, 3).to("cuda")
fullsize = ResizeTransform(0.5, 3).to("cuda")
integrate = VecInt(down_shape, 7).to("cuda")
integrate_64 = VecInt(down_shape_64, 7).to("cuda")

# Loss functions
similarity_loss = NormalizedCrossCorrelation().to("cuda")
smoothness_loss = Grad(penalty='l2')

# optimizer
optimizer = torch.optim.Adam(list(feature_extractor_training.parameters()) + list(scg_training.parameters()) +

                             list(upsampler1_training.parameters()) + list(upsampler2_training.parameters()) +
                             list(upsampler3_training.parameters()) + list(upsampler4_training.parameters()) +
                             list(upsampler5_training.parameters()) +

                             list(graph_layers1_training.parameters()) + list(graph_layers2_training.parameters()) +

                             list(conv_decoder1_training.parameters()) + list(conv_decoder2_training.parameters()) +
                             list(conv_decoder3_training.parameters()) +

                             list(conv_decoder4_training.parameters()) + list(conv_decoder5_training.parameters()) +
                             list(conv_decoder6_training.parameters()),

                             lr=lr)