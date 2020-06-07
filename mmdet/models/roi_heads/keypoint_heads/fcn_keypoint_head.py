import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn import functional as fn
from mmdet.models.builder import HEADS

from mmdet.core import auto_fp16, force_fp32, keypoint_target, keypoint_target2
from mmcv.cnn import ConvModule, build_upsample_layer

@HEADS.register_module
class FCNKeypointHead(nn.Module):
    def __init__(self,
                 num_convs=8,
                 roi_feat_size=28,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_keypoints=17,
                 heatmap_size=56,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 up_scale=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_keypoint=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0
                 )):
        super(FCNKeypointHead, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear", "carafe"'.format(
                    self.upsample_cfg['type']))
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor')
        self.num_keypoints = num_keypoints
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        #self.loss_keypoint = build_loss(loss_keypoint)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        upsample_cfg_.update(
            in_channels=upsample_in_channels,
            out_channels=num_keypoints,
            kernel_size=self.scale_factor*2,
            stride=self.scale_factor,
            padding = 1)

        #self.conv_logits = nn.Conv2d(self.conv_out_channels, num_keypoints, 1)
        self.upsample = build_upsample_layer(upsample_cfg_)
        #self.relu = nn.ReLU(inplace=True)
        self.up_scale = up_scale

    def init_weights(self):
        for m in [self.upsample]:
            if m is None:
                continue
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        # for m in self.convs:
        #     if m is None:
        #         continue
        #     else:
        #         nn.init.kaiming_normal_(
        #             m.conv.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.conv.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
        #x = self.conv_logits(x)
        x = fn.interpolate(x, size=None, scale_factor=self.up_scale,
                           mode="bilinear")
        return x

    def get_keypoints(self, keypoint_pred, det_bboxes,
                      ori_shape, scale_factor, rescale):
        heatmap_w = keypoint_pred.shape[3]
        heatmap_h = keypoint_pred.shape[2]

        num_preds, num_keypoints = keypoint_pred.shape[:2]

        if not rescale:
            scale_factor = 1.0

        bboxes = det_bboxes / scale_factor

        offset_x = bboxes[:, 0]
        offset_y = bboxes[:, 1]

        widths =  (bboxes[:, 2] - bboxes[:, 0]).clamp(min=1)
        heights = (bboxes[:, 3] - bboxes[:, 1]).clamp(min=1)

        width_corrections  = widths / heatmap_w
        height_corrections = heights / heatmap_h

        keypoints_idx = torch.arange(num_keypoints, device=keypoint_pred.device)
        xy_preds = torch.zeros((num_preds, num_keypoints, 4)).to(keypoint_pred.device)

        for i in range(num_preds):
            max_score, _ = keypoint_pred[i].view(num_keypoints, -1).max(1)
            max_score = max_score.view(num_keypoints, 1, 1)

            tmp_full_res = (keypoint_pred[i] - max_score).exp_()
            tmp_pool_res = (keypoint_pred[i] - max_score).exp_()
            roi_map_scores = tmp_full_res / tmp_pool_res.sum((1, 2), keepdim=True)

            pos = keypoint_pred[i].view(num_keypoints, -1).argmax(1)
            x_int = pos % heatmap_w
            y_int = (pos - x_int) // heatmap_w

            x = (x_int.float() + 0.5)*width_corrections[i]
            y = (y_int.float() + 0.5)*height_corrections[i]

            xy_preds[i, :, 0] = x + offset_x[i]
            xy_preds[i, :, 1] = y + offset_y[i]
            xy_preds[i, :, 2] = keypoint_pred[i][keypoints_idx, y_int, x_int]
            xy_preds[i, :, 3] = roi_map_scores[keypoints_idx, y_int, x_int]

        return xy_preds

    def get_keypoints2(self, origin_keypoint_pred, det_bboxes,
                      ori_shape, scale_factor, rescale):
        keypoint_pred = origin_keypoint_pred[:,:-1,:,:]
        return self.get_keypoints(keypoint_pred, det_bboxes, ori_shape, scale_factor, rescale)

    def select_sampling_results_with_keypoint(self, sampling_results, all_gt_keypoints_list):
        device = sampling_results[0].bboxes.device
        pos_proposals_list = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds_list = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        for i in range(len(pos_proposals_list)):
            pos_proposals = pos_proposals_list[i]
            pos_assigned_gt_inds = pos_assigned_gt_inds_list[i].cpu().numpy()
            num_pos = pos_proposals.size(0)
            filtered_pos_boxes = list()
            filtered_pos_gt_inds = list()
            gt_keypoints_list = all_gt_keypoints_list[i]
            if num_pos > 0:
                proposals_np = pos_proposals.cpu().numpy()
                for ii in range(num_pos):
                    gt_keypoints = gt_keypoints_list[pos_assigned_gt_inds[ii]].cpu().numpy()
                    gt_keypoints = gt_keypoints.reshape((-1, 3))
                    proposal = proposals_np[i]
                    x = gt_keypoints[...,0]
                    y = gt_keypoints[...,1]
                    vis = gt_keypoints[...,2] >= 1

                    kp_in_box = (x >= proposal[0]) & (x <= proposal[2]) & \
                                (y >= proposal[1]) & (y <= proposal[3])
                    select = (kp_in_box&vis).any()
                    #print(select)
                    if select:
                        filtered_pos_boxes.append(proposal)
                        filtered_pos_gt_inds.append(pos_assigned_gt_inds[ii])
                #print(filtered_pos_boxes)
                #print(filtered_pos_gt_inds)
            if len(filtered_pos_boxes) > 0:
                filtered_pos_boxes = np.stack(filtered_pos_boxes)
                filtered_pos_gt_inds = np.stack(filtered_pos_gt_inds)
                filtered_pos_boxes = torch.from_numpy(filtered_pos_boxes).to(device)
                filtered_pos_gt_inds = torch.from_numpy(filtered_pos_gt_inds).to(device)
            else:
                filtered_pos_boxes = torch.zeros((0, 4)).to(device)
                filtered_pos_gt_inds = torch.zeros((0)).to(device)
            sampling_results[i].pos_bboxes = filtered_pos_boxes
            sampling_results[i].pos_assigned_gt_inds = filtered_pos_gt_inds
        return sampling_results


    def get_target(self, sampling_results, gt_keypoints, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        keypoint_targets, valids = keypoint_target(pos_proposals, pos_assigned_gt_inds,
            gt_keypoints, rcnn_train_cfg)
        return keypoint_targets, valids.to(dtype=torch.bool)

    def get_target2(self, sampling_results, gt_keypoints, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        keypoint_targets = keypoint_target2(pos_proposals, pos_assigned_gt_inds,
            gt_keypoints, rcnn_train_cfg)
        return keypoint_targets

    @force_fp32(apply_to=('keypoint_pred', ))
    def loss(self, keypoint_pred, keypoint_targets, valids):
        loss = dict()
        valids = valids.nonzero().squeeze(1)
        N, K, H, W = keypoint_pred.shape
        keypoint_pred_logits = keypoint_pred.view(N*K, H*W)
        keypoint_pred_logits = keypoint_pred_logits[valids]

        keypoint_targets = keypoint_targets[valids].long()
        #print("<-------------------")
        #max_logits = keypoint_pred_logits.argmax(dim=1)
        # print("pred_logits")
        # print(max_logits)
        # print("target_logits")
        # print(keypoint_targets)
        #equals = (max_logits==keypoint_targets).squeeze(0).to(dtype=torch.long).sum()
        #print(equals)
        #print("------------------->")
        if keypoint_targets.shape[0] > 0:
            #loss['loss_keypoint'] = self.loss_keypoint(keypoint_pred_logits, keypoint_targets)
            loss_keypoint = fn.cross_entropy(keypoint_pred_logits, keypoint_targets, reduction="sum")
            loss_keypoint = loss_keypoint / valids.numel()
            loss['loss_keypoint'] = loss_keypoint
        else:
            loss['loss_keypoint'] = keypoint_pred_logits.sum()*0
        return loss


    @force_fp32(apply_to=('keypoint_pred', ))
    def loss2(self, keypoint_pred, keypoint_targets):
        loss = dict()
        batch_size = keypoint_pred.shape[0]*keypoint_targets.shape[1]
        keypoint_pred_logits = keypoint_pred.flatten()
        keypoint_targets_logits = keypoint_targets.flatten()
        mse_loss = fn.mse_loss(keypoint_pred_logits, keypoint_targets_logits, reduction="mean")*batch_size
        loss['loss_keypoint'] = mse_loss
        return loss
