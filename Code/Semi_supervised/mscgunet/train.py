from .model import *
from .losses import *
from .layers import *
import torch
import os


class Mscgunet:
    def __init__(self, device):
        self.lr = 1e-4
        self.range_flow = 7
        self.hyperparam1 = -1.2
        self.hyperparam2 = 1.0
        self.hyperparam3 = 1.0
        self.hyperparam4 = 10
        self.checkpoint_reload = True
        self.device = device

        # vector integrion to enforce diffeomorphic transform
        self.dim = 3
        self.int_downsize = 2
        self.down_shape = [int(dim / self.int_downsize) for dim in (32, 128, 128)]
        self.down_shape_64 = [int(dim / self.int_downsize) for dim in (16, 64, 64)]
        self.resize = ResizeTransform(2, 3).to(self.device)
        self.fullsize = ResizeTransform(0.5, 3).to(self.device)
        self.integrate = VecInt(self.down_shape, 7).to(self.device)
        self.integrate_64 = VecInt(self.down_shape_64, 7).to(self.device)

        # Loss functions
        self.similarity_loss = NormalizedCrossCorrelation().to(self.device)
        self.smoothness_loss = Grad(penalty='l2')

        # ========================================= Model Init - START =========================================================

        self.feature_extractor_training = Feature_Extractor(2, 3, 16).to(self.device)

        self.scg_training = SCG_block(in_ch=32, hidden_ch=9, node_size=(4, 4, 4)).to(self.device)

        self.upsampler1_training = convEncoder(9, 32, 32).to(self.device)
        self.upsampler2_training = convEncoder(32, 32, 32).to(self.device)
        self.upsampler3_training = convEncoder(32, 32, 32).to(self.device)
        self.upsampler4_training = convEncoder(32, 32, 32).to(self.device)
        self.upsampler5_training = convEncoder(16, 32, 32).to(self.device)

        self.conv_decoder1_training = convDecoder(33, 16).to(self.device)
        self.conv_decoder2_training = convDecoder(16, 16).to(self.device)
        self.conv_decoder3_training = convDecoder(16, 3).to(self.device)

        self.conv_decoder4_training = convDecoder(32, 16).to(self.device)
        self.conv_decoder5_training = convDecoder(16, 16).to(self.device)
        self.conv_decoder6_training = convDecoder(16, 3).to(self.device)

        self.graph_layers1_training = GCN_Layer(32, 16, bnorm=True, activation=nn.LeakyReLU(0.2), dropout=0.1).to(
            self.device)
        self.graph_layers2_training = GCN_Layer(16, 9, bnorm=True, activation=nn.LeakyReLU(0.2), dropout=0.1).to(
            self.device)

        self.stn_deformable = SpatialTransformer(size=(32, 128, 128), is_affine=False).to(self.device)
        self.stn_deformable_64 = SpatialTransformer(size=(16, 64, 64), is_affine=False).to(self.device)

        weight_xavier_init(self.graph_layers1_training, self.graph_layers2_training, self.scg_training)

        for param in self.stn_deformable.parameters():
            param.requires_grad = False
            param.volatile = True

    def initializeModel(self, model_dir):
        # load previous checkpoints
        checkpoint = torch.load(os.path.join(model_dir), map_location=self.device)

        self.feature_extractor_training.load_state_dict(checkpoint['feature_extractor_training'])
        self.scg_training.load_state_dict(checkpoint['scg_training'])

        self.graph_layers1_training.load_state_dict(checkpoint['graph_layers1_training'])
        self.graph_layers2_training.load_state_dict(checkpoint['graph_layers2_training'])

        self.upsampler1_training.load_state_dict(checkpoint["upsampler1_training"])
        self.upsampler2_training.load_state_dict(checkpoint["upsampler2_training"])
        self.upsampler3_training.load_state_dict(checkpoint["upsampler3_training"])
        self.upsampler4_training.load_state_dict(checkpoint["upsampler4_training"])
        self.upsampler5_training.load_state_dict(checkpoint["upsampler5_training"])

        self.conv_decoder1_training.load_state_dict(checkpoint["conv_decoder1_training"])
        self.conv_decoder2_training.load_state_dict(checkpoint["conv_decoder2_training"])
        self.conv_decoder3_training.load_state_dict(checkpoint["conv_decoder3_training"])

    def lossCal(self, CT, MRI, MRI_LBL):
        X = CT
        Y = MRI

        X = X.float().to(self.device)
        Y = Y.float().to(self.device)
        Ylbl = MRI_LBL.float().to(self.device)

        half_shape = tuple([i // 2 for i in X.shape[2:]])
        X_64 = F.interpolate(X, half_shape, mode='trilinear', align_corners=False)
        Y_64 = F.interpolate(Y, half_shape, mode='trilinear', align_corners=False)

        # ============================================ X-Y START ========================= ========================================

        enc_1_xy, enc_2_xy, enc_3_xy, enc_4_xy, enc_5_xy, enc_6_xy = self.feature_extractor_training(X, Y)
        A_xy, gx_xy, scg_loss_xy, z_hat_xy = self.scg_training(enc_6_xy)
        B_xy, C_xy, H_xy, W_xy, D_xy = enc_6_xy.size()
        gop_layers1_xy, A_layers1_xy = self.graph_layers1_training((gx_xy.reshape(B_xy, -1, C_xy), A_xy))
        gop_layers2_xy, A_layers2_xy = self.graph_layers2_training((gop_layers1_xy, A_layers1_xy))

        gop_layers2_xy = torch.bmm(A_layers2_xy, gop_layers2_xy)
        gop_layers2_xy = gop_layers2_xy + z_hat_xy

        # Upward trajectory
        gx_xy = gop_layers2_xy.reshape(B_xy, 9, 4, 4, 4)
        gx_xy = F.interpolate(gx_xy, (H_xy, W_xy, D_xy), mode='trilinear', align_corners=False)

        # Adding information from feature extractor directly to latent space info, this could provide a path for gradients to move faster
        gx_xy = self.upsampler1_training(gx_xy, enc_5_xy)  # 8
        gx_xy = self.upsampler2_training(gx_xy, enc_4_xy)  # 16
        gx_xy = self.upsampler3_training(gx_xy, enc_3_xy)  # 32
        gx_64_xy = self.upsampler4_training(gx_xy, enc_2_xy)  # 64
        gx_xy = self.upsampler5_training(gx_64_xy, enc_1_xy)  # 128, 32

        # Concat fixed image to final field before smoothening
        gx_xy = torch.cat((X, gx_xy), 1)

        # Conv decoder last three layers from voxel morph
        gx_xy = self.conv_decoder1_training(gx_xy)  # 128, 33 --> 16
        gx_xy = self.conv_decoder2_training(gx_xy)  # 128,16
        dvf_xy = self.conv_decoder3_training(gx_xy)  # 128,3

        # Conv decoder last three layers from voxel morph 64
        gx_64_xy = self.conv_decoder4_training(gx_64_xy)  # 128, 33 --> 16
        gx_64_xy = self.conv_decoder5_training(gx_64_xy)  # 128,16
        dvf_64_xy = self.conv_decoder6_training(gx_64_xy)  # 128,3

        # vector integration for diffeomorphic field
        pos_flow_xy = self.resize(dvf_xy)
        integrated_pos_flow_xy = self.integrate(pos_flow_xy)
        full_flow_xy = self.fullsize(integrated_pos_flow_xy)

        # vector integration  64
        pos_flow_64_xy = self.resize(dvf_64_xy)
        integrated_pos_flow_64_xy = self.integrate_64(pos_flow_64_xy)
        full_flow_64_xy = self.fullsize(integrated_pos_flow_64_xy)

        # warp
        fully_warped_image_xy = self.stn_deformable(Y, full_flow_xy)
        fully_warped_image_64_xy = self.stn_deformable_64(Y_64, full_flow_64_xy)

        # ============================================= X-Y END ====================================================================

        # ============================================= Y-X START ==================================================================

        enc_1_yx, enc_2_yx, enc_3_yx, enc_4_yx, enc_5_yx, enc_6_yx = self.feature_extractor_training(Y, X)
        A_yx, gx_yx, scg_loss_yx, z_hat_yx = self.scg_training(enc_6_yx)
        B_yx, C_yx, H_yx, W_yx, D_yx = enc_6_yx.size()
        gop_layers1_yx, A_layers1_yx = self.graph_layers1_training((gx_yx.reshape(B_yx, -1, C_yx), A_yx))
        gop_layers2_yx, A_layers2_yx = self.graph_layers2_training((gop_layers1_yx, A_layers1_yx))

        gop_layers2_yx = torch.bmm(A_layers2_yx, gop_layers2_yx)
        gop_layers2_yx = gop_layers2_yx + z_hat_yx

        # Upward trajectory
        gx_yx = gop_layers2_yx.reshape(B_yx, 9, 4, 4, 4)
        gx_yx = F.interpolate(gx_yx, (H_yx, W_yx, D_yx), mode='trilinear', align_corners=False)

        # Adding information from feature extractor directly to latent space info, this could provide a path for gradients to move faster
        gx_yx = self.upsampler1_training(gx_yx, enc_5_yx)  # 8
        gx_yx = self.upsampler2_training(gx_yx, enc_4_yx)  # 16
        gx_yx = self.upsampler3_training(gx_yx, enc_3_yx)  # 32
        gx_64_yx = self.upsampler4_training(gx_yx, enc_2_yx)  # 64
        gx_yx = self.upsampler5_training(gx_64_yx, enc_1_yx)  # 128, 32

        # Concat fixed image to final field before smoothening
        gx_yx = torch.cat((Y, gx_yx), 1)

        # Conv decoder last three layers from voxel morph
        gx_yx = self.conv_decoder1_training(gx_yx)  # 128, 33 --> 16
        gx_yx = self.conv_decoder2_training(gx_yx)  # 128,16
        dvf_yx = self.conv_decoder3_training(gx_yx)  # 128,3

        # Conv decoder last three layers from voxel morph 64
        gx_64_yx = self.conv_decoder4_training(gx_64_yx)  # 128, 33 --> 16
        gx_64_yx = self.conv_decoder5_training(gx_64_yx)  # 128,16
        dvf_64_yx = self.conv_decoder6_training(gx_64_yx)  # 128,3

        # vector integration for diffeomorphic field
        pos_flow_yx = self.resize(dvf_yx)
        integrated_pos_flow_yx = self.integrate(pos_flow_yx)
        full_flow_yx = self.fullsize(integrated_pos_flow_yx)

        # vector integration for diffeomorphic field 64
        pos_flow_64_yx = self.resize(dvf_64_yx)
        integrated_pos_flow_64_yx = self.integrate_64(pos_flow_64_yx)
        full_flow_64_yx = self.fullsize(integrated_pos_flow_64_yx)

        #  warp
        fully_warped_image_yx = self.stn_deformable(X, full_flow_yx)
        fully_warped_image_64_yx = self.stn_deformable_64(X_64, full_flow_64_yx)

        # ============================================= Y-X END ======================================================================

        cc_loss = self.similarity_loss(X, fully_warped_image_xy) + self.similarity_loss(Y, fully_warped_image_yx)
        cc_loss_64 = self.similarity_loss(X_64, fully_warped_image_64_xy) + self.similarity_loss(Y_64, fully_warped_image_64_yx)
        sm_loss = self.smoothness_loss.loss("", full_flow_xy) + self.smoothness_loss.loss("", full_flow_yx)
        sm_loss_64 = self.smoothness_loss.loss("", full_flow_64_xy) + self.smoothness_loss.loss("", full_flow_64_yx)
        scg_loss = scg_loss_xy + scg_loss_yx
        total_loss = self.hyperparam1 * cc_loss + self.hyperparam3 * sm_loss + self.hyperparam2 * scg_loss + cc_loss_64 * -0.5 + sm_loss_64 * 0.5

        ################################################################################################

        psuedo_lbl = self.stn_deformable(Ylbl, full_flow_xy)

        return total_loss, fully_warped_image_xy, psuedo_lbl
