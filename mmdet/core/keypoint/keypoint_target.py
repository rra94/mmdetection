import numpy as np
import torch
import math

def keypoint_target(pos_proposals_list, pos_assigned_gt_inds_list, all_gt_keypoints_list,
                cfg):
    #cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    keypoint_targets = []
    valids = []
    for i in range(len(pos_proposals_list)):
        keypoint_target, valid = keypoint_target_single(pos_proposals_list[i],
                                                        pos_assigned_gt_inds_list[i],
                                                        all_gt_keypoints_list[i], cfg)
        keypoint_targets.append(keypoint_target)
        valids.append(valid)
    keypoint_targets = torch.cat(list(keypoint_targets))
    valids = torch.cat(list(valids))
    return keypoint_targets, valids

def keypoint_target_single(pos_proposals, pos_assigned_gt_inds, gt_keypoints_list, cfg):
    heatmap_size = cfg.heatmap_size
    num_pos = pos_proposals.size(0)
    num_keypoints = cfg.num_keypoints
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        offset_x_list = proposals_np[:, 0]
        offset_y_list = proposals_np[:, 1]
        scale_x_list = heatmap_size / (proposals_np[:, 2] - proposals_np[:, 0])
        scale_y_list = heatmap_size / (proposals_np[:, 3] - proposals_np[:, 1])
        targets = []
        valids = []
        for i in range(num_pos):
            gt_keypoints = gt_keypoints_list[pos_assigned_gt_inds[i]].cpu().numpy()
            gt_keypoints = gt_keypoints.reshape((-1, 3))

            proposal = proposals_np[i]

            x = gt_keypoints[...,0]
            y = gt_keypoints[...,1]

            x_boundary_inds = x == proposal[2]
            y_boundary_inds = y == proposal[3]

            offset_x = offset_x_list[i]
            offset_y = offset_y_list[i]
            scale_x =  scale_x_list[i]
            scale_y = scale_y_list[i]

            x = (x - offset_x)*scale_x
            x = np.floor(x).astype(np.int64)
            y = (y - offset_y)*scale_y
            y = np.floor(y).astype(np.int64)

            x[x_boundary_inds] = heatmap_size - 1
            y[y_boundary_inds] = heatmap_size - 1

            valid_loc = (x>=0) & \
                        (y>=0) & \
                        (x<heatmap_size) & \
                        (y<heatmap_size)

            vis = gt_keypoints[..., 2] > 0
            valid = (valid_loc & vis).astype(np.int64)
            lin_ind = y * heatmap_size + x
            #lin_ind = y * heatmap_size - y*heatmap_size
            heatmap = lin_ind*valid
            targets.append(heatmap)
            valids.append(valid)
        keypoint_targets = torch.from_numpy(np.concatenate(targets)).to(pos_proposals.device)
        valids = torch.from_numpy(np.concatenate(valids)).to(pos_proposals.device)
    else:
        keypoint_targets = torch.zeros((0, num_keypoints)).to(pos_proposals.device)
        valids = torch.zeros((0, num_keypoints)).to(pos_proposals.device)
    #print(keypoint_targets)
    #print(valids)
    return keypoint_targets, valids


def keypoint_target2(pos_proposals_list, pos_assigned_gt_inds_list, all_gt_keypoints_list,
                cfg):
    #cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    keypoint_targets = []
    for i in range(len(pos_proposals_list)):
        keypoint_target = keypoint_target_single2(pos_proposals_list[i],
                                                        pos_assigned_gt_inds_list[i],
                                                        all_gt_keypoints_list[i], cfg)
        keypoint_targets.append(keypoint_target)
    keypoint_targets = torch.cat(list(keypoint_targets))
    return keypoint_targets

def keypoint_target_single2(pos_proposals, pos_assigned_gt_inds, gt_keypoints_list, cfg):
    heatmap_size = cfg.heatmap_size
    num_pos = pos_proposals.size(0)
    num_keypoints = cfg.num_keypoints
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        offset_x_list = proposals_np[:, 0]
        offset_y_list = proposals_np[:, 1]
        scale_x_list = heatmap_size / (proposals_np[:, 2] - proposals_np[:, 0])
        scale_y_list = heatmap_size / (proposals_np[:, 3] - proposals_np[:, 1])
        # targets = []
        # valids = []
        targets = []
        for i in range(num_pos):
            gt_keypoints = gt_keypoints_list[pos_assigned_gt_inds[i]].cpu().numpy()
            gt_keypoints = gt_keypoints.reshape((-1, 3))

            proposal = proposals_np[i]

            x = gt_keypoints[...,0]
            y = gt_keypoints[...,1]

            x_boundary_inds = x == proposal[2]
            y_boundary_inds = y == proposal[3]

            offset_x = offset_x_list[i]
            offset_y = offset_y_list[i]
            scale_x =  scale_x_list[i]
            scale_y = scale_y_list[i]

            x = (x - offset_x)*scale_x
            x = np.floor(x).astype(np.int64)
            y = (y - offset_y)*scale_y
            y = np.floor(y).astype(np.int64)

            x[x_boundary_inds] = heatmap_size - 1
            y[y_boundary_inds] = heatmap_size - 1

            valid_loc = (x>=0) & \
                        (y>=0) & \
                        (x<heatmap_size) & \
                        (y<heatmap_size)

            vis = gt_keypoints[..., 2] > 0
            valid = (valid_loc & vis).astype(np.int64)
            keypoint_maps = np.zeros(shape=(num_keypoints+1, heatmap_size, heatmap_size), dtype=np.float32)  # +1 for bg
            for id in range(num_keypoints):
                if valid[id] == 0:
                    continue
                x_d, y_d = x[id], y[id]
                add_gaussian(keypoint_maps[id], x_d, y_d, 4, 5)
            keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
            targets.append(keypoint_maps)
        keypoint_targets = torch.from_numpy(np.concatenate(targets)).to(pos_proposals.device)
        return keypoint_targets

def add_gaussian(keypoint_map, x, y, stride, sigma):
    n_sigma = 4
    tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
    tl[0] = max(tl[0], 0)
    tl[1] = max(tl[1], 0)

    br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
    map_h, map_w = keypoint_map.shape
    br[0] = min(br[0], map_w * stride)
    br[1] = min(br[1], map_h * stride)

    shift = stride / 2 - 0.5
    for map_y in range(tl[1] // stride, br[1] // stride):
        for map_x in range(tl[0] // stride, br[0] // stride):
            d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                 (map_y * stride + shift - y) * (map_y * stride + shift - y)
            exponent = d2 / 2 / sigma / sigma
            if exponent > 4.6052:  # threshold, ln(100), ~0.01
                continue
            keypoint_map[map_y, map_x] += math.exp(-exponent)
            if keypoint_map[map_y, map_x] > 1:
                keypoint_map[map_y, map_x] = 1