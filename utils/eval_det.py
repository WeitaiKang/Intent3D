# ------------------------------------------------------------------------
# Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

""" Generic Code for Object Detection Evaluation

    Input:
    For each class:
        For each image:
            Predictions: box, score
            Groundtruths: box
    
    Output:
    For each class:
        precision-recal and average precision
    
    Author: Charles R. Qi
    
    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
"""
import numpy as np
import torch
import ipdb
from tqdm import tqdm
st = ipdb.set_trace

VALID_TEST_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 79, 80, 81, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 111, 113, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 129, 131, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 144, 146, 147, 148, 150, 151, 152, 154, 155, 156, 157, 158, 160, 161, 163, 165, 166, 167, 168, 172, 174, 175, 177, 178, 179, 181, 182, 183, 185, 187, 188, 191, 192, 193, 194, 195, 198, 201, 203, 204, 205, 206, 208, 209, 210, 214, 215, 216, 218, 222, 223, 225, 230, 231, 234, 235, 237, 238, 240, 241, 250, 255, 256, 258, 260, 262, 263, 264, 267, 270, 276, 280, 282, 284, 285, 286, 288, 290, 304, 306, 312, 313, 314, 315, 321, 323, 324, 325, 326, 327, 328, 329, 330, 331, 333, 334, 335, 337, 338, 339, 341, 342, 343, 344, 346, 347, 348, 349, 350, 351, 353, 354, 355, 357, 360, 361, 363, 364, 368, 369, 371, 372, 373, 381, 384, 389, 396, 398, 402, 413, 414, 422, 426, 427, 446, 455]

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from metric_util import calc_iou  # axis-aligned 3D box IoU


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def corners_to_ends(box):
    min_xyz = torch.min(box, axis=0)[0]
    max_xyz = torch.max(box, axis=0)[0]
    return torch.cat((min_xyz, max_xyz))


def _volume_par(box):
    return (box[:, 3] - box[:, 0]) * (box[:, 4] - box[:, 1]) * (box[:, 5] - box[:, 2])


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return torch.clamp(xB - xA, 0) * torch.clamp(yB - yA, 0) * torch.clamp(zB - zA, 0)


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union


def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    '''
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    iou = torch.zeros((N,M)).to(boxes1.device)
    union = torch.zeros((N,M)).to(boxes1.device)
    for n in range(N):
        for m in range(M):
            iou[n,m], union[n,m] = _iou3d(boxes1[n], boxes2[m])
    '''
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume


def get_iou(bb1, bb2):
    """ Compute IoU of two bounding boxes.
        ** Define your bod IoU function HERE **
    """
    # pass
    iou3d = calc_iou(bb1, bb2)
    return iou3d


from box_util import box3d_iou


def get_iou_obb(bb1, bb2):
    iou3d, iou2d = box3d_iou(bb1, bb2)
    return iou3d


def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)


def eval_intention_ap(pred, gt, ovthresh=0.25, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = gt[img_id]
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence) # (num_allpred_of_allsample, )
    BB = np.array(BB) # (num_allpred_of_allsample, 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence) # descending order
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids) # num_allpred_of_allsample
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    print('begin to eval intention ap')
    for d in tqdm(range(nd)):
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        # print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos + 1e-8)

    # print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return voc_ap(rec, prec)