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
"""Main script for language modulation."""

import os

import numpy as np
import torch
import torch.distributed as dist
local_rank = int(os.environ.get('LOCAL_RANK', 0))  # Default to 0 if not set

from main_utils import parse_option, BaseTrainTester
from src.joint_det_dataset import Joint3DDataset
from src.grounding_evaluator import GroundingEvaluator
from models import BeaUTyDETR
from models import APCalculator

from tqdm import tqdm
import datetime

import ipdb
st = ipdb.set_trace

class TrainTester(BaseTrainTester):

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""

        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        print('Loading datasets:', sorted(list(dataset_dict.keys())))

        if args.eval:
            train_dataset = None
        else:
            train_dataset = Joint3DDataset(
                dataset_dict=dataset_dict,
                test_dataset=args.test_dataset,
                split='train' if not args.debug else 'val',
                use_color=args.use_color,
                overfit=args.debug,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                butd=args.butd,
                augment_det=args.augment_det
            )
        
        test_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='val' if not args.eval_test else 'test',
            use_color=args.use_color,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            butd=args.butd,
        )
        return train_dataset, test_dataset

    # Initialize the model.
    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19
        model = BeaUTyDETR(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls,
            pointnet_ckpt=args.pp_checkpoint,
            data_path = args.data_root,
            self_attend=args.self_attend
        )
        return model

    # input data.
    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(), # ([B, 50000, 6]) xyz + colour
            'text': batch_data['utterances'],                   # list[B]  text
            'point_instance_label': batch_data['point_instance_label'], # ([B, 50000])  instance label
            "det_boxes": batch_data['all_detected_boxes'],      # ([B, 132, 6]) groupfree detection boxes
            "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],  # ([B, 132]) mask
            "det_class_ids": batch_data['all_detected_class_ids']   # ([B, 132])  box id
        }


    # only eval one epoch.
    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        # 7 layers: proposal, last, 0-4
        prefixes = ['last_']

        evaluator = GroundingEvaluator(
            only_root=False, thresholds=[0.25, 0.5],     # TODO only_root=True
            prefixes=prefixes
        )

        ap_calculator_list = [
            APCalculator(iou_thresh)
            for iou_thresh in args.ap_iou_thresholds # [0.25, 0.50]
        ]

        # NOTE Main eval branch
        test_loader = tqdm(test_loader)
        for batch_idx, batch_data in enumerate(test_loader):
            # note forward and compute loss
            stat_dict, end_points = self._main_eval_branch(     
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            if evaluator is not None:
                for prefix in prefixes:
                    # note only consider the last layer
                    if prefix != 'last_':
                        continue

                    # Intent3D evaluation
                    batch_pred_box, batch_pred_conf, batch_gt, batch_gt_mask = \
                        evaluator.evaluate(end_points, prefix)  
                    batch_pred_box_allgpu = gather_tensor(batch_pred_box)
                    batch_pred_conf_allgpu = gather_tensor(batch_pred_conf)
                    batch_gt_allgpu = gather_tensor(batch_gt)
                    batch_gt_mask_allgpu = gather_tensor(batch_gt_mask)
                        
                    if dist.get_rank() == 0:
                        # evaluate AP
                        for ap_calculator in ap_calculator_list:
                            ap_calculator.step(batch_pred_box_allgpu, batch_pred_conf_allgpu,
                                               batch_gt_allgpu, batch_gt_mask_allgpu)

        evaluator.synchronize_between_processes()
        APs = []
        if dist.get_rank() == 0:
            # evaluate Top1-Acc
            c_25 = evaluator.dets[("last_", 0.25, 'top1_acc')] / max(evaluator.gts[("last_", 0.25, 'top1_acc')], 1)
            c_50 = evaluator.dets[("last_", 0.5,  'top1_acc')] / max(evaluator.gts[("last_", 0.5,  'top1_acc')], 1)

            if not (args.eval_val or args.eval_test):
                self.tensorboard.item["val_score"]["top1_acc_0.25"] = c_25
                self.tensorboard.item["val_score"]["top1_acc_0.5"] = c_50
            print('=====================>', f'{prefixes[0]} Top1-Acc: 0.25: {c_25}, 0.5: {c_50}', '<=====================')

            if not (args.eval_val or args.eval_test):
                # tensorboard
                for key in self.tensorboard.item["val_loss"]:
                    self.tensorboard.item["val_loss"][key] = stat_dict[key] / len(test_loader)
                self.tensorboard.dump_tensorboard("val_loss", epoch)

            for i, ap_calculator in enumerate(ap_calculator_list):
                ap = ap_calculator.compute_metrics()
                self.logger.info(
                    '=====================>'
                    f'{prefixes[0]} IOU THRESH: {args.ap_iou_thresholds[i]}, AP: {ap}'
                    '<====================='
                )
                APs.append(ap)

            if not (args.eval_val or args.eval_test):
                # tensorboard
                for i in range(len(args.ap_iou_thresholds)):
                    self.tensorboard.item["val_score"][f"AP_{args.ap_iou_thresholds[i]}"] = APs[i]
                self.tensorboard.dump_tensorboard("val_score", epoch)

        return APs

def is_dist_avail_and_initialized():
    """
    Returns:
        True if distributed training is enabled
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    """
    Returns:
        The number of processes in the process group
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def gather_tensor(data):
    if is_dist_avail_and_initialized():
        tensor_list = [torch.empty_like(data) for _ in range(get_world_size())]
        dist.all_gather(tensor_list, data)
        gathered_tensor = torch.cat(tensor_list, dim=0)
    else:
        gathered_tensor = data
    return gathered_tensor

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

if __name__ == '__main__':
    # huggingface
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    opt = parse_option()    
    
    # distributed 
    # torch.cuda.set_device(opt.local_rank)
    torch.cuda.set_device(local_rank)
    setup_for_distributed(local_rank == 0)

    # https://github.com/open-mmlab/mmcv/issues/1969#issuecomment-1304721237
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))  
    
    # cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(opt)
    _ = train_tester.main(opt)
