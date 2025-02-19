# ------------------------------------------------------------------------
# Modification: EDA
# Created: 05/21/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA 
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""A class to collect and evaluate language grounding results."""

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
import utils.misc as misc
import numpy as np

def softmax(x):
    """Numpy function for softmax."""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

# Evaluator
class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, only_root=False, thresholds=[0.25, 0.5], prefixes=[], filter_non_gt_boxes=False):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = thresholds
        self.prefixes = prefixes
        self.filter_non_gt_boxes = filter_non_gt_boxes
        self.reset()

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = {
            (prefix, t, mode): 0
            for prefix in self.prefixes
            for t in self.thresholds
            for mode in ['top1_acc']
        }
        self.gts = dict(self.dets)

    def print_stats(self):
        """Print accumulated accuracies."""
        mode_str = {
            'bbf': 'semantic alignment'
        }
        for prefix in self.prefixes:
            for mode in ['bbf']:
                for t in self.thresholds:
                    print(
                        prefix, mode_str[mode], 'Acc%.2f:' % t,
                        ', '.join([
                            'Top-%d: %.5f' % (
                                k,
                                self.dets[(prefix, t, k, mode)]
                                / max(self.gts[(prefix, t, k, mode)], 1)
                            )
                            for k in self.topks
                        ])
                    )

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)

        if misc.is_main_process():
            merged_predictions = {}
            for key in all_dets[0].keys():
                merged_predictions[key] = 0
                for p in all_dets:
                    merged_predictions[key] += p[key]
            self.dets = merged_predictions

            merged_predictions = {}
            for key in all_gts[0].keys():
                merged_predictions[key] = 0
                for p in all_gts:
                    merged_predictions[key] += p[key]
            self.gts = merged_predictions

    def evaluate(self, end_points, prefix):
        """
        Evaluate bounding box IoU by semantic alignment.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)

        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        
        # step compute similarity between vision and text
        proj_tokens = end_points['proj_tokens']             # text feature   (B, L, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # vision feature (B, 256, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))  # similarity ([B, 256, L]) 
        sem_scores_ = (sem_scores / 0.07).softmax(-1)                           # softmax ([B, 256, L])
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256) # ([B, 256, 256])
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_ # ([B, P=256, L=256])

        # Highest scoring box -> iou
        batch_gt = []
        batch_gt_mask = []
        batch_pred_box = []
        batch_pred_conf = []
        for bid in range(len(positive_map)):
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            # you don't know the exact number of gt in evaluation. 
            # use the first pos to retrieve 5 predictions, since the 1:num_obj are the same
            pmap = positive_map[bid, [0]]
            scores_main = (
                sem_scores[bid].unsqueeze(0)  # (1, Q=256, 256)
                * pmap.unsqueeze(1)  # (1, 1, 256)
            ).sum(-1)  # (1, Q=256)

            # total score
            scores = scores_main # (1, Q=256)

            # 10 predictions per gt box
            top = scores.argsort(1, True) # the index of descending
            pbox = pred_bbox[bid, top.reshape(-1)] # (Q=256, 6)

            # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (num_gt_obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (Q, 6)
            )  # (num_gt_obj, Q)

            # calculate the top1 acc
            self.accumulated_top1_acc(ious)

            # accumulated pred and gt boxes for AP
            batch_gt.append(gt_bboxes[bid][:5])
            batch_gt_mask.append(end_points['box_label_mask'][bid][:5])
            batch_pred_box.append(pred_bbox[bid])
            batch_pred_conf.append(scores.reshape(-1))

        return torch.stack(batch_pred_box), torch.stack(batch_pred_conf), \
            torch.stack(batch_gt), torch.stack(batch_gt_mask)

    def accumulated_top1_acc(self, ious):
        for prefix in self.prefixes:
            for threshold in self.thresholds:  # [0.25, 0.5]
                self.dets[(prefix, threshold, 'top1_acc')] += (ious[:, 0] >= threshold).any().int().item()
                self.gts[(prefix, threshold, 'top1_acc')] += 1

    # Get the postion label of the decoupled text component.
    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])                  # main
        positive_map[positive_map > 0] = 1                      
        gt_center = end_points['center_label'][:, :, 0:3]       
        gt_size = end_points['size_gts']                        
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)     # GT box cxcyczwhd
        return positive_map, gt_bboxes