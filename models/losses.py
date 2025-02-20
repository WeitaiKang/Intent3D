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
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def _volume_par(box):
    return (
        (box[:, 3] - box[:, 0])
        * (box[:, 4] - box[:, 1])
        * (box[:, 5] - box[:, 2])
    )


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
        torch.clamp(xB - xA, 0)
        * torch.clamp(yB - yA, 0)
        * torch.clamp(zB - zA, 0)
    )


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union

# 3DIoU loss
def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all(), print(boxes1)
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all(), print(boxes2)
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.

    This class is taken from Group-Free code.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Args:
            gamma: Weighting parameter for hard and easy examples.
            alpha: Weighting parameter for positive and negative examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """
        PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
        max(x, 0) - x * z + log(1 + exp(-abs(x))) in

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0) - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def forward(self, input, target, weights):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss
        loss = loss.squeeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()      # B, K
    seed_xyz = end_points['seed_xyz']               # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']   # B, 1, K
    gt_center = end_points['center_label'][:, :, :3]            # B, G=132, 3
    gt_size = end_points['size_gts'][:, :, :3]                  # B, G, 3
    B = gt_center.shape[0]  # batch size
    K = seed_xyz.shape[1]   # number if points from p++ output  1024
    G = gt_center.shape[1]  # number of gt boxes (with padding) 132

    # Assign each point to a GT object
    point_instance_label = end_points['point_instance_label']           # B, num_points=5000
    obj_assignment = torch.gather(point_instance_label, 1, seed_inds)   # B, K=1024
    obj_assignment[obj_assignment < 0] = G - 1                          # bg points to last gt
    obj_assignment_one_hot = torch.zeros((B, K, G)).to(seed_xyz.device)
    obj_assignment_one_hot.scatter_(2, obj_assignment.unsqueeze(-1), 1)

    # Normalized distances of points and gt centroids
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, G, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)       # (B, K, G, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxG
    euclidean_dist1 = (
        euclidean_dist1 * obj_assignment_one_hot
        + 100 * (1 - obj_assignment_one_hot)
    )  # BxKxG
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()

    # Find the points that lie closest to each gt centroid
    topk_inds = (
        torch.topk(euclidean_dist1, topk, largest=False)[1]
        * box_label_mask[:, :, None]
        + (box_label_mask[:, :, None] - 1)
    )  # BxGxtopk
    topk_inds = topk_inds.long()  # BxGxtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, Gxtopk
    batch_inds = torch.arange(B)[:, None].repeat(1, G*topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([
        batch_inds,
        topk_inds
    ], -1).view(-1, 2).contiguous()

    # Topk points closest to each centroid are marked as true objects
    objectness_label = torch.zeros((B, K + 1)).long().to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)
    objectness_label[objectness_label_mask < 0] = 0 

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(
        seeds_obj_cls_logits.view(B, K, 1),
        objectness_label.unsqueeze(-1),
        weights=cls_weights
    )
    objectness_loss = cls_loss_src.sum() / B

    return objectness_loss


class HungarianMatcher(nn.Module):
    """
    Assign targets to predictions.

    This class is taken from MDETR and is modified for our purposes.

    For efficiency reasons, the [targets don't include the no_object].
    Because of this, in general, there are [more predictions than targets].
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2,
                 soft_token=False):
        """
        Initialize matcher.

        Args:
            cost_class: relative weight of the classification error
            cost_bbox: relative weight of the L1 bounding box regression error
            cost_giou: relative weight of the giou loss of the bounding box
            soft_token: whether to use soft-token prediction
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        self.soft_token = soft_token

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform the matching.

        Args:
            outputs: This is a dict that contains at least these entries:
                "pred_logits" (tensor): [batch_size, num_queries, num_classes]
                "pred_boxes" (tensor): [batch_size, num_queries, 6], cxcyczwhd
            targets: list (len(targets) = batch_size) of dict:
                "labels" (tensor): [num_target_boxes]
                    (where num_target_boxes is the no. of ground-truth objects)
                "boxes" (tensor): [num_target_boxes, 6], cxcyczwhd
                "positive_map" (tensor): [num_target_boxes, 256]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Notation: {B: batch_size, Q: num_queries, C: num_classes}
        bs, num_queries = outputs["pred_logits"].shape[:2]  # Q: num_queries = 256

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Q, C=256]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 6]

        # Also concat the target labels and boxes
        positive_map = torch.cat([t["positive_map"] for t in targets])  # (B * num_gt_each_B, 256)
        tgt_ids = torch.cat([v["labels"] for v in targets]) # (B * num_gt_each_B,)
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # (B * num_gt_each_B, 6)

        if self.soft_token: # True
            # pad if necessary
            if out_prob.shape[-1] != positive_map.shape[-1]:
                positive_map = positive_map[..., :out_prob.shape[-1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1))  # (B*(Q=256), B * num_gt_each_B)
        else:
            # Compute the classification cost.
            # Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching,
            # it can be ommitted. DETR
            # out_prob = out_prob * out_objectness.view(-1, 1)
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)    # (B*(Q=256), B * num_gt_each_B)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou3d(     # # (B*(Q=256), B * num_gt_each_B)
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )

        # Final cost matrix
        C = (                               # # (B, Q=256, B * num_gt_each_B)
            self.cost_bbox * cost_bbox          
            + self.cost_class * cost_class     
            + self.cost_giou * cost_giou        
        ).view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets] # [num_gt_each_B, num_gt_each_B, ...]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ] # i: 0, 1, 2, ..., B-1, c: (B, Q=256, num_gt_each_B), c[i]: (Q=256, num_gt_each_B)
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),  # matched pred boxes
                torch.as_tensor(j, dtype=torch.int64)  # corresponding gt boxes
            )
            for i, j in indices
        ] # len(indices) = B, len(indices[i]) = 2 (one for pred, one for gt), len(indices[i][0]) = num_gt_each_B

# Compute loss
class SetCriterion(nn.Module):
    def __init__(self, matcher, losses={}, eos_coef=0.1, temperature=0.07):
        """
        Parameters:
            matcher: module that matches targets and proposals
            losses: list of all the losses to be applied
            eos_coef: weight of the no-object category
            temperature: used to sharpen the contrastive logits
        """
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef    # 0.1
        self.losses = losses
        self.temperature = temperature
    
    def loss_labels_st(self, outputs, targets, indices, num_boxes, adapt_w=None):
        """Soft token prediction (with objectness)."""
        logits = outputs["pred_logits"].log_softmax(-1)  # (B, Q, 256)
        positive_map = torch.cat([t["positive_map"] for t in targets])

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        # Labels, by default lines map to the last element, no_object
        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        # Compute entropy
        entropy = 0 # torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef
        diff = loss_ce.sum(dim = -1)
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        diff =  diff / diff.amax()
        return losses, diff
    
  
    # object detection loss.
    def loss_boxes(self, outputs, targets, indices, num_boxes, adapt_w=None):
        """Compute bbox losses."""
        assert 'pred_boxes' in outputs
        if adapt_w is not None:
            adapt_w = adapt_w.view(-1, 1).repeat(1, outputs['pred_boxes'].shape[1])
        idx = self._get_src_permutation_idx(indices) # len(idx) = 2, len(idx[0]) = len(idx[1]) = B * num_gt_each_B
        if adapt_w is not None:
            adapt_w = adapt_w[idx]
        src_boxes = outputs['pred_boxes'][idx] # (B, Q=256, 6)[B_idx, src_idx] --> (B * num_gt_each_B, 6)
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0) # (B * num_gt_each_B, 6)
        
        loss_bbox = (
            F.l1_loss(
                src_boxes[..., :3], target_boxes[..., :3],
                reduction='none'
            )
            + 0.2 * F.l1_loss(
                src_boxes[..., 3:], target_boxes[..., 3:],
                reduction='none'
            )
        ) # (B * num_gt_each_B, 3)
        losses = {}
        
        loss_giou = 1 - torch.diag(generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes))) # (B * num_gt_each_B,)

        if adapt_w is not None:
            losses['loss_bbox'] = (loss_bbox * adapt_w.unsqueeze(-1)).sum() / num_boxes
            losses['loss_giou'] = (loss_giou * adapt_w).sum() / num_boxes
        else:
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses, None

    def loss_contrastive_align(self, outputs, targets, indices, num_boxes, adapt_w=None):
        """Compute contrastive losses between projected queries and tokens."""
        tokenized = outputs["tokenized"]

        # Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens, dim
        norm_img_emb = outputs["proj_queries"]  # B, num_queries, dim
        logits = (
            torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
            / self.temperature
        )  # B, num_queries, num_tokens

        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap[..., :logits.shape[-1]]
        positive_map = positive_map > 0

        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits

        # Loss 1: which tokens should each query match?
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)
        nb_pos = positive_map.sum(2) + 1e-6
        entropy = 0 # -torch.log(nb_pos+1e-6) / nb_pos  # entropy of 1/nb_pos
        box_to_token_loss_ = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~boxes_with_pos, 0)
        diff = (box_to_token_loss_ * mask).sum(dim = -1)
        if adapt_w is not None:
            box_to_token_loss = ((box_to_token_loss_ * mask).sum(dim = -1) * adapt_w).sum()
        else:
            box_to_token_loss = (box_to_token_loss_ * mask).sum()

        # Loss 2: which queries should each token match?
        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)
        nb_pos = positive_map.sum(1) + 1e-6
        entropy = 0 # -torch.log(nb_pos+1e-6) / nb_pos
        token_to_box_loss = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        diff = diff + (token_to_box_loss * tmask).sum(dim = -1)
        if adapt_w is not None:
            token_to_box_loss = ((token_to_box_loss * tmask).sum(dim = -1) * adapt_w).sum()
        else:
            token_to_box_loss = (token_to_box_loss * tmask).sum()

        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        diff =  diff / diff.amax()
        return {"loss_sem_align": tot_loss / num_boxes}, diff
    
    def loss_verb_obj(self, outputs, targets, indices, num_boxes, adapt_w=None):
        """Compute contrastive losses between projected queries and tokens."""
        tokenized = outputs["tokenized"]
        inds = tokenized['attention_mask'].sum(1) - 1
        pmap = torch.cat([
            t['target_vo_map'][i] for t, (_, i) in zip(targets, indices)
            ], dim=0) # B*num_gt_each_B, 256
        max_num_vo = pmap.max() - 0.5
        # assert max_num_vo % 1 == 0, f"max_num_vo: {max_num_vo}"
        max_num_vo = int(max_num_vo)
        vo_map = torch.stack([t['target_vo_map'][0] for t in targets], dim=0) # B, 256
        len_text = outputs["v_text"].shape[1] # B, L, dim
        vo_query = outputs["vo_query"] # B, num_queries, dim    
        o_text = F.normalize(outputs["o_text"], p=2, dim=-1) # B, L, dim

        tot_loss = 0
        diff = 0
        # Given each object modulated query, 
        # we align positive query with its verb, and align negative query with "not mentioned".
        for v_idx in range(0, max_num_vo+1):
            v_mask = (vo_map == v_idx)[:, :len_text] # B, L
            if v_mask.sum() == 0:
                continue
            # B, dim
            v_mask_ = (v_mask.sum(dim=-1, keepdim=True) > 0).float() * v_mask + \
                (v_mask.sum(dim=-1, keepdim=True) == 0).float() * torch.ones_like(v_mask)
            v_text = (outputs["v_text"] * v_mask_.unsqueeze(-1)).sum(dim = 1) / v_mask_.sum(dim=-1, keepdim=True).clamp(min=1) 
            vo_query = F.normalize(vo_query * v_text.unsqueeze(1), p=2, dim=-1)

            logits = (
                torch.matmul(vo_query, o_text.transpose(-1, -2))
                / self.temperature
            )  # B, num_queries, num_text_tokens

            # construct a map such that positive_map[k, i, j] = True
            # iff query i is associated to token j in batch item k
            current_pmap = (pmap == (v_idx + 0.5))
            positive_map = torch.zeros(logits.shape, device=logits.device)
            # handle 'not mentioned'
            positive_map[torch.arange(len(inds)), :, inds] = 0.5
            positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
            positive_map_ = positive_map.clone()
            # handle true mentions
            idx = self._get_src_permutation_idx(indices)
            positive_map_[idx] = current_pmap[..., :logits.shape[-1]].float()
            positive_map_ += (positive_map_.sum(dim=-1, keepdim=True)==0) * positive_map
            positive_map = positive_map_ > 0

            valid_mask = v_mask.any(dim=-1, keepdim=True) # B, 1
            # Mask for matches <> 'not mentioned'
            mask = torch.full(
                logits.shape[:2],
                self.eos_coef,
                dtype=torch.float32, device=logits.device
            )
            mask[idx] = 1.0
            mask *= valid_mask
            # Token mask for matches <> 'not mentioned'
            tmask = torch.full(
                (len(logits), logits.shape[-1]),
                self.eos_coef,
                dtype=torch.float32, device=logits.device
            )
            tmask[torch.arange(len(inds)), inds] = 1.0
            tmask *= valid_mask

            # Positive logits are those who correspond to a match
            positive_logits = -logits.masked_fill(~positive_map, 0)
            negative_logits = logits

            # Loss 1: which tokens should each query match?
            boxes_with_pos = positive_map.any(2)
            pos_term = positive_logits.sum(2)
            neg_term = negative_logits.logsumexp(2)
            nb_pos = positive_map.sum(2) + 1e-6
            box_to_token_loss_ = (
                (pos_term / nb_pos + neg_term)
            ).masked_fill(~boxes_with_pos, 0)
            diff = diff + (box_to_token_loss_ * mask).sum(dim = -1)
            if adapt_w is not None:
                box_to_token_loss = ((box_to_token_loss_ * mask).sum(dim = -1) * adapt_w).sum()
            else:
                box_to_token_loss = (box_to_token_loss_ * mask).sum()

            # Loss 2: which queries should each token match?
            tokens_with_pos = positive_map.any(1)
            pos_term = positive_logits.sum(1)
            neg_term = negative_logits.logsumexp(1)
            nb_pos = positive_map.sum(1) + 1e-6
            token_to_box_loss = (
                (pos_term / nb_pos + neg_term)
            ).masked_fill(~tokens_with_pos, 0)
            diff = diff + (token_to_box_loss * tmask).sum(dim = -1)
            if adapt_w is not None:
                token_to_box_loss = ((token_to_box_loss * tmask).sum(dim = -1) * adapt_w).sum()
            else:
                token_to_box_loss = (token_to_box_loss * tmask).sum()

            tot_loss += (box_to_token_loss + token_to_box_loss) / 2
        diff =  diff / diff.amax()
        return {"loss_verb_obj": tot_loss / num_boxes}, diff

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        """
        Perform the loss computation.

        Parameters:
             outputs: dict of tensors
             targets: list of dicts, such that len(targets) == batch_size.
        """
        # STEP Retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets)

        num_boxes = sum(len(inds[1]) for inds in indices) # (B * num_gt_each_B,)
        
        # Compute all the requested losses
        losses = {}
        loss_dict, diff = self.loss_labels_st(outputs, targets, indices, num_boxes, None)
        adapt_w = diff.sigmoid() + 0.5
        losses.update(loss_dict)

        loss_dict, diff = self.loss_contrastive_align(outputs, targets, indices, num_boxes, adapt_w)
        adapt_w = diff.sigmoid() + 0.5
        losses.update(loss_dict)

        loss_dict, diff = self.loss_verb_obj(outputs, targets, indices, num_boxes, adapt_w)
        adapt_w = diff.sigmoid() + 0.5
        losses.update(loss_dict)

        loss_dict, _ = self.loss_boxes(outputs, targets, indices, num_boxes, adapt_w)
        losses.update(loss_dict)

        return losses, indices

# loss
def compute_hungarian_loss(end_points, num_decoder_layers, set_criterion,
                           query_points_obj_topk=5):
    """Compute Hungarian matching loss containing CE, bbox and giou."""
    prefixes = ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    prefixes = ['proposal_'] + prefixes     # 6+1: 'proposal_'  'last_' '0head_'  '1head_'  '2head_'  '3head_'  '4head_'

    # target GT box
    gt_center = end_points['center_label'][:, :, 0:3]
    gt_size = end_points['size_gts']
    gt_labels = end_points['sem_cls_label']
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)
    # text
    positive_map = end_points['positive_map']               # main obj. positive_map
    box_label_mask = end_points['box_label_mask']           # (132,) target object mask
    vo_map = end_points['target_vo_map'] # B, MAX_NUM_OBJ, 256

    target = [
        {
            "labels": gt_labels[b, box_label_mask[b].bool()], # not used
            "boxes": gt_bbox[b, box_label_mask[b].bool()],
            "positive_map": positive_map[b, box_label_mask[b].bool()],
            "target_vo_map": vo_map[b, box_label_mask[b].bool()],
        }
        for b in range(gt_labels.shape[0])
    ]

    loss_ce, loss_bbox, loss_giou, loss_sem_align, loss_vo = 0, 0, 0, 0, 0
    for i, prefix in enumerate(prefixes):
        output = {}
        if 'proj_tokens' in end_points:
            output['proj_tokens'] = end_points['proj_tokens']           
            output['proj_queries'] = end_points[f'{prefix}proj_queries']
            output['tokenized'] = end_points['tokenized']
            output['vo_query'] = end_points[f'{prefix}vo_query']
            output['v_text'] = end_points['v_text']
            output['o_text'] = end_points['o_text']
        
        # Get predicted boxes and labels
        pred_center = end_points[f'{prefix}center']     # B, K, 3
        pred_size = end_points[f'{prefix}pred_size']    # (B,K,3) (l,w,h)
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        pred_logits = end_points[f'{prefix}sem_cls_scores']     # (B, Q, n_class)
        output['pred_logits'] = pred_logits
        output["pred_boxes"] = pred_bbox

        # NOTE Compute all the requested losses, forward
        losses, indices = set_criterion(output, target)
        for loss_key in losses.keys():
            end_points[f'{prefix}_{loss_key}'] = losses[loss_key]
        loss_ce += losses.get('loss_ce', 0)
        loss_bbox += losses['loss_bbox']
        loss_giou += losses.get('loss_giou', 0)
        if 'proj_tokens' in end_points:
            loss_sem_align += losses['loss_sem_align']
            loss_vo += losses['loss_verb_obj']

    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk
        )
    else:
        query_points_generation_loss = 0.0

    # total loss
    tgt_obj_cls_loss = bce(end_points)
    tgt_obj_cls_loss = tgt_obj_cls_loss*20
    query_points_generation_loss = 8 * query_points_generation_loss
    loss_ce = 1.0 / (num_decoder_layers + 1) * loss_ce              
    loss_bbox = 1.0 / (num_decoder_layers + 1) * 5 * loss_bbox
    loss_giou = 1.0 / (num_decoder_layers + 1) * loss_giou
    loss_sem_align = 1.0 / (num_decoder_layers + 1) * loss_sem_align              
    loss_vo = 1.0 / (num_decoder_layers + 1) * loss_vo 

    # weight
    # loss_giou = loss_giou * 5
    # query_points_generation_loss = query_points_generation_loss * 10
    # loss_vo = loss_vo * 0.5
    loss = query_points_generation_loss + loss_ce + loss_bbox + loss_giou + \
                    loss_sem_align + loss_vo + tgt_obj_cls_loss

    end_points['loss_ce'] = loss_ce
    end_points['loss_bbox'] = loss_bbox
    end_points['loss_giou'] = loss_giou
    end_points['loss_query_points_generation'] = query_points_generation_loss
    end_points['loss_sem_align'] = loss_sem_align
    end_points['loss_vo'] = loss_vo
    end_points['loss_tgt_obj_cls'] = tgt_obj_cls_loss
    end_points['loss'] = loss
    return loss, end_points

def bce(end_points):
    total_loss = 0
    for logits in end_points['vclue_tgt_cls_logits_list']:
        sigmoid_logits = torch.sigmoid(logits)

        losses = - (end_points['correct_proposal'].float() * torch.log(sigmoid_logits + 1e-6) + 
                    (1 - end_points['correct_proposal'].float()) * torch.log(1 - sigmoid_logits + 1e-6))

        mask = (end_points['all_detected_bbox_label_mask'] == True) & (end_points['correct_proposal'] != -100)
        weighted_losses = losses * mask.float()

        loss_sum = weighted_losses.sum()

        total_loss += loss_sum / (mask.sum().float() + 1e-6)

    return total_loss