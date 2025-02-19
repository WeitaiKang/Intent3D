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
"""Dataset and data loader."""

import json
import sys, os
sys.path.append(os.getcwd())
import spacy
from tqdm import tqdm
from six.moves import cPickle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast


from data.model_util_scannet import ScannetDatasetConfig
from data.scannet_utils import read_label_mapping
from src.visual_data_handlers import Scan


NUM_CLASSES = 485
DC = ScannetDatasetConfig(NUM_CLASSES)
DC18 = ScannetDatasetConfig(18)
MAX_NUM_OBJ = 132

sys.path.append(".")
sys.path.append('..')
from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
MAX_NUM_SEG = 132

class Joint3DDataset(Dataset):
    """Dataset utilities for Intent3D."""

    def __init__(self, dataset_dict={'scanintend': 1},
                 test_dataset='scanintend',
                 split='train', overfit=False,
                 data_path='./',
                 use_color=False, 
                 detect_intermediate=False,
                 butd=False, augment_det=False):
        """Initialize dataset (here for Intent3D utterances)."""
        self.dataset_dict = dataset_dict
        self.test_dataset = test_dataset
        self.split = split
        self.use_color = use_color
        self.overfit = overfit
        self.detect_intermediate = detect_intermediate
        self.augment = self.split == 'train'
        self.data_path = data_path
        self.visualize = False  # manually set this to True to debug
        self.augment_det = augment_det

        self.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
        
        # semantic label
        self.label_map = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='id'
        )
        self.label_map18 = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40id'
        )
        self.label_mapclass = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40class'
        )   

        # transformer tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        print('Loading %s files, take a breath!' % split)
        # load train/val_v3scans.pkl
        if split == 'test':
            self.scans = unpickle_data(f'{self.data_path}/val_v3scans.pkl')
        else:
            self.scans = unpickle_data(f'{self.data_path}/{split}_v3scans.pkl')
        self.scans = list(self.scans)[0]

        # load text dataset
        self.nlp = spacy.load("en_core_web_sm")
        self.annos = self.load_annos()

    # load text data
    def load_annos(self):
        """Load annotations"""
        annos = self.load_scanintend_annos()
        return annos
    
    def load_scanintend_annos(self):
        """Load annotations of scanintend."""

        ###########################
        # If you have done text decoupling ahead #
        ###########################
        with open(os.path.join(self.data_path, 'intention_sentence', f'{self.split}.json')) as f:
            reader = json.load(f)
        annos = reader

        ###########################
        # otherwise, use text decoupling on raw intention sentences by `python prepare_data.py` with the following lines#
        ###########################
        # with open(os.path.join(self.data_path, 'intention_sentence', f'{self.split}_samples_dict_vg_format_clean_duplicate.json')) as f:
        #     reader = json.load(f)
        # parse_verb_obj_I(annos, self.tokenizer, self.nlp)
        # save_path = os.path.join(self.data_path, 'intention_sentence', f'{self.split}.json')
        # with open(save_path, 'w') as f:
        #     json.dump(annos, f)
        # print(f'----- Save {save_path}')

        return annos

    # point cloud augmentation
    def _augment(self, pc, color=None, rotate=True):
        augmentations = {}

        # Rotate/flip only if we don't have a view_dep sentence
        if rotate:
            theta_z = 90*np.random.randint(0, 4) + (2*np.random.rand() - 1) * 5
            # Flipping along the YZ plane
            augmentations['yz_flip'] = np.random.random() > 0.5
            if augmentations['yz_flip']:
                pc[:, 0] = -pc[:, 0]
            # Flipping along the XZ plane
            augmentations['xz_flip'] = np.random.random() > 0.5
            if augmentations['xz_flip']:
                pc[:, 1] = -pc[:, 1]
        else:
            theta_z = (2*np.random.rand() - 1) * 5
        augmentations['theta_z'] = theta_z
        pc[:, :3] = rot_z(pc[:, :3], theta_z)
        # Rotate around x
        theta_x = (2*np.random.rand() - 1) * 2.5
        augmentations['theta_x'] = theta_x
        pc[:, :3] = rot_x(pc[:, :3], theta_x)
        # Rotate around y
        theta_y = (2*np.random.rand() - 1) * 2.5
        augmentations['theta_y'] = theta_y
        pc[:, :3] = rot_y(pc[:, :3], theta_y)

        # Add noise
        noise = np.random.rand(len(pc), 3) * 5e-3
        augmentations['noise'] = noise
        pc[:, :3] = pc[:, :3] + noise

        # Translate/shift
        augmentations['shift'] = np.random.random((3,))[None, :] - 0.5
        pc[:, :3] += augmentations['shift']

        # Scale
        augmentations['scale'] = 0.98 + 0.04*np.random.random()
        pc[:, :3] *= augmentations['scale']

        # Color
        if color is not None:
            color += self.mean_rgb
            color *= 0.98 + 0.04*np.random.random((len(color), 3))
            color -= self.mean_rgb
        return pc, color, augmentations
    
    # get point clouds
    def _get_pc(self, scan):
        """Return a point cloud representation of current scene."""
        # Color
        color = None
        if self.use_color:
            color = scan.color - self.mean_rgb

        # Augmentations
        augmentations = {}
        if self.split == 'train' and self.augment:
            pc, color, augmentations = self._augment(scan.pc, color)
            scan.pc = pc

        # Concatenate representations
        point_cloud = scan.pc
        if color is not None:
            point_cloud = np.concatenate((point_cloud, color), 1)

        return point_cloud, augmentations

    # get GT Box.
    def _get_target_boxes(self, anno, scan):
        """Return gt boxes to detect."""
        bboxes = np.zeros((MAX_NUM_OBJ, 6))
        assert isinstance(anno['target_id'], list)
        tids = anno['target_id']
        point_instance_label = -np.ones(len(scan.pc))
        for t, tid in enumerate(tids):
            point_instance_label[scan.three_d_objects[tid]['points']] = t
        
        bboxes[:len(tids)] = np.stack([
            scan.get_object_bbox(tid).reshape(-1) for tid in tids
        ])
        bboxes = np.concatenate((
            (bboxes[:, :3] + bboxes[:, 3:]) * 0.5,
            bboxes[:, 3:] - bboxes[:, :3]
        ), 1)
        if self.split == 'train' and self.augment:  # jitter boxes
            bboxes[:len(tids)] *= (0.95 + 0.1*np.random.random((len(tids), 6)))
        bboxes[len(tids):, :3] = 1000
        
        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[:len(tids)] = 1
        
        return bboxes, box_label_mask, point_instance_label

    # GroupFree detection boxes
    def _get_detected_objects(self, split, scan_id, augmentations):
        # Initialize
        all_detected_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_detected_bbox_label_mask = np.array([False] * MAX_NUM_OBJ)
        detected_class_ids = np.zeros((MAX_NUM_OBJ,))

        # Load
        if split == 'train':
            detected_dict = np.load(
                f'{self.data_path}/group_free_pred_bboxes/group_free_pred_bboxes_{split}/{scan_id}.npy',
                allow_pickle=True
            ).item()
        else:
            try:
                detected_dict = np.load(
                    f'{self.data_path}/group_free_pred_bboxes/group_free_pred_bboxes_val/{scan_id}.npy',
                    allow_pickle=True
                ).item()
            except:
                detected_dict = np.load(
                    f'{self.data_path}/group_free_pred_bboxes/group_free_pred_bboxes_test/{scan_id}.npy',
                    allow_pickle=True
                ).item()

        all_bboxes_ = np.array(detected_dict['box'])
        classes = detected_dict['class']
        cid = np.array([DC.nyu40id2class[
            self.label_map[c]] for c in detected_dict['class']
        ])
        all_bboxes_ = np.concatenate((
            (all_bboxes_[:, :3] + all_bboxes_[:, 3:]) * 0.5,
            all_bboxes_[:, 3:] - all_bboxes_[:, :3]
        ), 1)

        assert len(classes) < MAX_NUM_OBJ
        assert len(classes) == all_bboxes_.shape[0]

        num_objs = len(classes)
        all_detected_bboxes[:num_objs] = all_bboxes_
        all_detected_bbox_label_mask[:num_objs] = np.array([True] * num_objs)
        detected_class_ids[:num_objs] = cid
        # Match current augmentations
        if self.augment and self.split == 'train':
            all_det_pts = box2points(all_detected_bboxes).reshape(-1, 3)
            all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])
            all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])
            all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])
            if augmentations.get('yz_flip', False):
                all_det_pts[:, 0] = -all_det_pts[:, 0]
            if augmentations.get('xz_flip', False):
                all_det_pts[:, 1] = -all_det_pts[:, 1]
            all_det_pts += augmentations['shift']
            all_det_pts *= augmentations['scale']
            all_detected_bboxes = points2box(all_det_pts.reshape(-1, 8, 3))
        
        if self.augment_det and self.split == 'train':
            min_ = all_detected_bboxes.min(0)
            max_ = all_detected_bboxes.max(0)
            rand_box = (
                (max_ - min_)[None]
                * np.random.random(all_detected_bboxes.shape)
                + min_
            )
            corrupt = np.random.random(len(all_detected_bboxes)) > 0.7
            all_detected_bboxes[corrupt] = rand_box[corrupt]
            detected_class_ids[corrupt] = np.random.randint(
                0, len(DC.nyu40ids), (len(detected_class_ids))
            )[corrupt]
        return (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids
        )

    # data
    def __getitem__(self, index):
        """Get current batch for input index."""
        split = self.split

        # Read annotation and point clouds
        anno = self.annos[index]
        scan = self.scans[anno['scan_id']]
        scan.pc = np.copy(scan.orig_pc)

        # point cloud
        point_cloud, augmentations = self._get_pc(scan)

        # ground truth boxes
        gt_bboxes, box_label_mask, point_instance_label = \
            self._get_target_boxes(anno, scan)

        # Intent3D
        target_vo_map, target_v_map = self._get_verb_obj_by_parse(anno)
        # groupfree Detected boxes
        (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids
        ) = self._get_detected_objects(split, anno['scan_id'], augmentations)
        obj_boxes = all_detected_bboxes
        obj_mask = all_detected_bbox_label_mask
        ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(torch.from_numpy(gt_bboxes[:int(box_label_mask.sum())])),  # (num_gt_obj, 6)
                box_cxcyczwhd_to_xyzxyz(torch.from_numpy(obj_boxes[:int(obj_mask.sum())]))  # (num_box, 6)
            )  # (num_gt_obj, num_box)
        ious = ious.numpy()
        correct_proposal = np.ones((MAX_NUM_SEG,)) * -100
        correct_proposal[:int(obj_mask.sum())] = (ious.max(0) >= 0.25) * 1.0

        # Return
        _labels = np.zeros(MAX_NUM_OBJ)
        ret_dict = {
            'box_label_mask': box_label_mask.astype(np.float32),
            'center_label': gt_bboxes[:, :3].astype(np.float32),
            'sem_cls_label': _labels.astype(np.int64), # not used
            'size_gts': gt_bboxes[:, 3:].astype(np.float32),
            'correct_proposal': correct_proposal.astype(np.float32),
            "scan_ids": anno['scan_id'],
            "point_clouds": point_cloud.astype(np.float32),
            "utterances": anno['utterance'] + ' . not mentioned',
            # NOTE text component position label
            "target_vo_map": target_vo_map.astype(np.float32),
            "positive_map": target_v_map.astype(np.float32),
            "point_instance_label": point_instance_label.astype(np.int64),
            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64),
        }
        return ret_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)

    def _get_verb_obj_by_parse(self, anno):
        num_gt_obj = len(anno['target_id'])
        verb_obj_tokenIdx = anno['verb_obj_roberta_span'] # list of [idx, [verb start, verb end, object start, object end]]
        vo_map = torch.zeros((256,), dtype=torch.float) 
        v_map = torch.zeros((256,), dtype=torch.float)
        for v_o_i in verb_obj_tokenIdx:
            idx, v_o = v_o_i
            vs, ve, os, oe = v_o
            vo_map[vs:ve].fill_(idx)
            vo_map[os:oe].fill_(idx + 0.5)
            v_map[vs:ve].fill_(1)

        # NOTE text position label # multiTgt
        target_vo_map = np.zeros((MAX_NUM_OBJ, 256))
        target_vo_map[:num_gt_obj] = vo_map[None, ...]
        
        target_v_map = np.zeros((MAX_NUM_OBJ, 256))
        v_map = v_map / (v_map.sum(-1) + 1e-12)
        v_map = v_map.numpy()
        target_v_map[:num_gt_obj] = v_map[None, ...]
        return target_vo_map, target_v_map

def rot_x(pc, theta):
    """Rotate along x-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [1.0, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_y(pc, theta):
    """Rotate along y-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1.0, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_z(pc, theta):
    """Rotate along z-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0]
        ]),
        pc.T
    ).T

def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack((
        np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1)
    ), axis=1)

def points2box(box):
    """Convert vertices (Nx8x3) to box center/hwd coordinates (Nx6)."""
    return np.concatenate((
        (box.min(1) + box.max(1)) / 2,
        box.max(1) - box.min(1)
    ), axis=1)


# Save all scans to pickle.
def save_data(filename, split, data_path):
    """Save all scans to pickle."""
    import multiprocessing as mp

    # Read all scan files
    scan_path = data_path + 'scans/'
    with open('data/meta_data/scannetv2_%s.txt' % split) as f:
        scan_ids = [line.rstrip() for line in f]    # train/val scene id list.
    print('{} scans found.'.format(len(scan_ids)))

    # Load data
    n_items = len(scan_ids)
    n_processes = 4  # min(mp.cpu_count(), n_items)
    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)
    all_scans = dict()
    iter_obj = [
        (scan_id, scan_path)
        for scan_id in scan_ids
    ]

    for i, data in enumerate(
        pool.imap(scannet_loader, iter_obj, chunksize=chunks)
    ):
        all_scans[scan_ids[i]] = data
    pool.close()
    pool.join()

    # Save data
    print('pickle time')
    pickle_data(filename, all_scans)

# load scannet
def scannet_loader(iter_obj):
    """Load the scans in memory, helper function."""
    scan_id, scan_path = iter_obj
    print(scan_id)
    return Scan(scan_id, scan_path, True)

def pickle_data(file_name, *args):
    """Use (c)Pickle to save multiple objects in a single file."""
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()

# read from pkl
def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()

def parse_verb_obj_I(annos, tokenizer, nlp):

    # import SpaCy model
    print('Begin text decoupling......')
    for anno in tqdm(annos):
        anno['utterance'] = anno['utterance'].replace('-', ' ')
        text = anno['utterance']
        doc = nlp(text)

        # tokenize by RobertaTokenizerFast
        encoded_input = tokenizer(text, return_offsets_mapping=True)
        offset_mapping = encoded_input['offset_mapping']
        
        # analysis of verb and object by SpaCy
        all_verb_obj = []
        verb_obj_list = []
        verb_obj_idx = 0
        # find sentence beginning with "I"
        for token in doc:
            if token.text.lower() == "i":
                i_subject = token
                break

        for token in doc:
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ == "dobj" or child.dep_ == "pobj":
                        direct_obj = child
                        # Check if there is an object connected by a preposition
                        for grandchild in child.children:
                            if grandchild.dep_ == "prep" and child.text not in ["after", "before", "during"]:
                                for grandgrandchild in grandchild.children:
                                    if grandgrandchild.dep_ == "pobj":
                                        direct_obj = grandgrandchild  # Update the object to the object of a preposition
                        verb_obj_idx += 1
                        verb_obj_list.append((verb_obj_idx, token, direct_obj))
                        break
                    elif child.dep_ == "prep" and child.text not in ["after", "before", "during"]:
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                direct_obj = grandchild
                                verb_obj_idx += 1
                                verb_obj_list.append((verb_obj_idx, token, direct_obj))
                                break
        
        # If no collocation is added (meaning no collocation is found) and the sentence begins with "I"
        if len(verb_obj_list) == 0:
            # For each verb, use the "I" at the beginning of the sentence as the object
            for token in doc:
                if token.pos_ == "VERB":
                    verb_obj_idx += 1
                    verb_obj_list.append((verb_obj_idx, token, i_subject))

        # Mapping to the position of tokenized text
        assert len(verb_obj_list) > 0, f"verb_obj_list is None, {text}"
        for idx, verb, obj in verb_obj_list:
            verb_start, verb_end = verb.idx, verb.idx + len(verb.text)

            # Calculate the start and end index
            verb_token_start_index = next((i for i, offset in enumerate(offset_mapping) if offset[0] == verb_start), None)
            assert verb_token_start_index is not None, f"verb_token_start_index is None, {verb_start}, {text}"

            verb_token_end_index = next((i for i, offset in enumerate(offset_mapping) if offset[1] == verb_end), None)

            # if verb_token_start_index is not None:
            vstart = verb_token_start_index
            if verb_token_end_index is not None:
                vend = max(verb_token_end_index, vstart+1)
            else:
                vend = vstart+1

            obj_start, obj_end = obj.idx, obj.idx + len(obj.text)

            # Calculate the start and end index
            obj_token_start_index = next((i for i, offset in enumerate(offset_mapping) if offset[0] == obj_start), None)
            assert obj_token_start_index is not None, f"obj_token_start_index is None, {obj_start}, {text}"
            obj_token_end_index = next((i for i, offset in enumerate(offset_mapping) if offset[1] == obj_end), None)

            ostart = obj_token_start_index
            if obj_token_end_index is not None:
                oend = max(obj_token_end_index, ostart+1)
            else:
                oend = ostart+1
            all_verb_obj.append((idx, (vstart, vend, ostart, oend)))                

        assert len(all_verb_obj) > 0
        anno['verb_obj_roberta_span'] = all_verb_obj
    
    print('End text decoupling!')