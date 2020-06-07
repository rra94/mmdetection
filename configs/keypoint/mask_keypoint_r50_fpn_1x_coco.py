_base_ = [
    '../_base_/models/mask_keypoint_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_keypoint.py',
    '../_base_/schedules/schedule_1x_0.01.py',
    '../_base_/default_runtime.py'
]