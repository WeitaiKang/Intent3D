# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Helper functions to calculate Average Precisions for 3D object detection."""

import numpy as np
import torch

from utils.eval_det import eval_intention_ap, get_iou
from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from utils.box_util import get_3d_box

import ipdb
st = ipdb.set_trace


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def flip_axis_to_camera(pc):
    """
    Flip X-right, Y-forward, Z-up to X-right, Y-down, Z-forward.

    Input and output are both (N, 3) array
    """
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def flip_axis_to_depth(pc):
    """Inverse of flip_axis_to_camera."""
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def softmax(x):
    """Numpy function for softmax."""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def sigmoid(x):
    """Numpy function for sigmoid."""
    s = 1 / (1 + np.exp(-x))
    return s


def parse_predictions(end_points, config_dict, prefix="", size_cls_agnostic=False):
    """ Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}
    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    pred_center = end_points[f'{prefix}center']  # (B,num_proposal=256,3)
    # pred_heading_class = torch.argmax(end_points[f'{prefix}heading_scores'], -1)  # B,num_proposal
    # pred_heading_residual = torch.gather(end_points[f'{prefix}heading_residuals'], 2,
    #                                      pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    # pred_heading_residual.squeeze_(2)

    if size_cls_agnostic:
        pred_size = end_points[f'{prefix}pred_size']  # (B, num_proposal, 3)
    else:
        pred_size_class = torch.argmax(end_points[f'{prefix}size_scores'], -1)  # B,num_proposal
        pred_size_residual = torch.gather(end_points[f'{prefix}size_residuals'], 2,
                                          pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                             3))  # B,num_proposal,1,3
        pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points[f'{prefix}sem_cls_scores'][..., :-1], -1)  # B,num_proposal
    sem_cls_probs = softmax(end_points[f'{prefix}sem_cls_scores'].detach().cpu().numpy())  # softmax, B,num_proposal,19

    num_proposal = pred_center.shape[1]     # 256
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    # pred_size_check = end_points[f'{prefix}pred_size']  # B,num_proposal,3
    # pred_bbox_check = end_points[f'{prefix}bbox_check']  # B,num_proposal,3

    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = 0  #config_dict['dataset_config'].class2angle( \
            #     pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            if size_cls_agnostic:
                box_size = pred_size[i, j].detach().cpu().numpy()
            else:
                box_size = config_dict['dataset_config'].class2size( \
                    int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())
            
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------
    if config_dict.get('hungarian_loss', False):
        # obj_logits = np.zeros(pred_center[:,:,None,0].shape) + 5 # (B,K,1)
        # obj_logits[end_points[f'{prefix}indices']] = 5
        if f'{prefix}objectness_scores' in end_points:
            obj_logits = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()
            obj_prob = sigmoid(obj_logits)  # (B,K)
        else: 
            obj_prob = (1 - sem_cls_probs[:,:,-1])
            sem_cls_probs = sem_cls_probs[..., :-1] / obj_prob[..., None]
    else:
        obj_logits = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()
        obj_prob = sigmoid(obj_logits)[:, :, 0]  # (B,256)
    
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 2] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    # 3D NMS
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            # assert (len(pick) > 0)
            if len(pick) > 0:
                pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points[f'{prefix}pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class):
                # if config_dict.get('hungarian_loss', False) and ii == config_dict['dataset_config'].num_class - 1:
                #    continue
                try:
                    cur_list += [
                        (ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j])
                        for j in range(pred_center.shape[1])
                        if pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']
                    ]
                except:
                    st()
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i, j].item(), pred_corners_3d_upright_camera[i, j], obj_prob[i, j]) \
                                       for j in range(pred_center.shape[1]) if
                                       pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])

    return batch_pred_map_cls


def parse_groundtruths(end_points, config_dict, size_cls_agnostic):
    """
    Parse groundtruth labels to OBB parameters.

    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
            [
                [(gt_sem_cls, gt_box_params)_j for j in range(n_obj[i])]
                for i in range(B)
            ]
    """
    center_label = end_points['center_label']
    if size_cls_agnostic:
        size_gts = end_points['size_gts']
    else:
        size_class_label = end_points['size_class_label']
        size_residual_label = end_points['size_residual_label']
    box_label_mask = end_points['box_label_mask']
    sem_cls_label = end_points['sem_cls_label']
    bsize = center_label.shape[0]

    K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i, j] == 0:
                continue
            heading_angle = 0
            if size_cls_agnostic:
                box_size = size_gts[i, j].detach().cpu().numpy()
            else:
                box_size = config_dict['dataset_config'].class2size(int(size_class_label[i, j].detach().cpu().numpy()),
                                                                    size_residual_label[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i, j, :])
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([
            (sem_cls_label[i, j].item(), gt_corners_3d_upright_camera[i, j])
            for j in range(gt_corners_3d_upright_camera.shape[1])
            if box_label_mask[i, j] == 1
        ])
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls


class APCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=0.25):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.reset()

    def step(self, batch_pred_box_allgpu, batch_pred_conf_allgpu,
                batch_gt_allgpu, batch_gt_mask_allgpu):
        """ Accumulate one batch of prediction and groundtruth.
        batch_pred_box_allgpu: bs, num_proposal, 6
        batch_pred_conf_allgpu: bs, num_proposal
        batch_gt_allgpu: bs, max_num_obj, 6
        batch_gt_mask_allgpu: bs, max_num_obj
        """

        bsize = len(batch_pred_box_allgpu)
        assert (bsize == len(batch_gt_allgpu))
        for i in range(bsize):
            # Filtering ground truth boxes where mask is 1
            gt_mask = batch_gt_mask_allgpu[i].to(torch.bool)
            gt_boxes = batch_gt_allgpu[i][gt_mask].cpu().numpy()
            self.gt_map[self.scan_cnt] = gt_boxes

            # Prepare predictions in the format {img_id: [(bbox, score), ...]}
            self.pred_map[self.scan_cnt] = [(bbox, score) for bbox, score in 
                                            zip(batch_pred_box_allgpu[i].cpu().numpy(),
                                                batch_pred_conf_allgpu[i].cpu().numpy())]
            
            self.scan_cnt += 1

    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
                    # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (num_gt_obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (Q, 6)
            )  # (num_gt_obj, Q)
        """
        ap = eval_intention_ap(self.pred_map, self.gt_map, ovthresh=self.ap_iou_thresh, get_iou_func = get_iou)
        return ap

    def reset(self):
        self.gt_map = {}  # {scan_id: [bbox, bbox, ...], ...}
        self.pred_map = {}  # {scan_id: [(bbox, score), (bbox, score), ...], ...}
        self.scan_cnt = 0