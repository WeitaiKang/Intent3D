import os
import sys
import time
import numpy as np

import torch

from tensorboardX import SummaryWriter  # tensorboard --logdir=./output/tensorboard --port 3090

class TensorBoard():
    def __init__(self, tensorboard_path, distributed_rank):

        self.init_log_item()

        if distributed_rank == 0:
            self.tensorboard_writer = SummaryWriter(tensorboard_path)
    
    def dump_tensorboard(self, phase, timestamp):
        log = {
            # phase:[value]
            "loss": ["loss", "loss_bbox", "loss_ce", "loss_sem_align", "loss_giou",
                     "loss_query_points_generation", 'loss_vo', "loss_tgt_obj_cls"],
            "score": ["top1_acc_0.25", "top1_acc_0.5", "AP_0.25", "AP_0.5"],
            "learning_rate": ["lr_base", "lr_pointnet"]
        }

        # train loss
        if phase == "train_loss":
            for val in log["loss"]:
                self.tensorboard_writer.add_scalar(
                    "{}/{}".format("train_loss", val),
                    self.item[phase][val],
                    timestamp
                )
        
        # lr
        if phase == "train_lr":
            for val in log["learning_rate"]:
                self.tensorboard_writer.add_scalar(
                    "{}/{}".format("learning_rate", val),
                    self.item[phase][val],
                    timestamp
                )
        
        # val loss
        if phase == "val_loss":
            for val in log["loss"]:
                self.tensorboard_writer.add_scalar(
                    "{}/{}".format("val_loss", val),
                    self.item[phase][val],
                    timestamp
                )

        # val score
        if phase == "val_score":
            for val in log["score"]:
                self.tensorboard_writer.add_scalar(
                    "{}/{}".format("val_score", val),
                    self.item[phase][val],
                    timestamp
                )

    def init_log_item(self):
        self.item = {
            "train_lr":{
                "lr_base": [],
                "lr_pointnet": [],
            },

            "train_loss":{
                "loss":[],
                "loss_bbox": [],
                "loss_ce": [],
                "loss_sem_align": [],
                "loss_giou": [],
                "loss_query_points_generation": [],
                'loss_vo': [],
                'loss_tgt_obj_cls': [],
            },

            "val_score":{
                "top1_acc_0.25": [],
                "top1_acc_0.5": [],
                "AP_0.25": [],
                "AP_0.5": []
            },

            "val_loss":{
                "loss":[],
                "loss_bbox": [],
                "loss_ce": [],
                "loss_sem_align": [],
                "loss_giou": [],
                "loss_query_points_generation": [],
                'loss_vo': [],
                'loss_tgt_obj_cls': [],
            }
        }