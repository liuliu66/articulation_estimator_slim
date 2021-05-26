import os, sys

import torch
import torch.nn as nn

from .pointnet2 import PointNet2
from .estimation_head import EstimationHead


class ArticulationEstimator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 n_max_parts=7):
        super(ArticulationEstimator, self).__init__()
        self.n_max_parts = n_max_parts

        self.backbone = PointNet2(in_channels)

        self.nocs_head = EstimationHead(n_max_parts, mixed_pred=True)

    def forward(self, return_loss=True, **input):
        if return_loss:
            return self.forward_train(**input)
        else:
            return self.forward_test(**input)

    def forward_train(self, **input):
        P = input['pts']
        if 'pts_feature' in input.keys():
            P_feature = input['pts_feature']
        else:
            P_feature = None

        feat, feat_encode = self.backbone(P, P_feature)

        pred_dict = self.nocs_head(feat, feat_encode)
        loss_result = self.nocs_head.loss(pred_dict, mode='train', **input)

        return loss_result

    def forward_test(self, **input):
        P = input['pts']
        if 'pts_feature' in input.keys():
            P_feature = input['pts_feature']
        else:
            P_feature = None

        if P.dim() == 2:
            P = P.unsqueeze(0)
        if P_feature.dim() == 2:
            P_feature = P_feature.unsqueeze(0)
        feat, feat_encode = self.backbone(P, P_feature)
        pred_dict = self.nocs_head(feat, feat_encode)

        return pred_dict

    @property
    def with_nocs(self):
        return hasattr(self, 'nocs_head') and self.nocs_head is not None

