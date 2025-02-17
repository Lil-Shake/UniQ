import h5py
import json
import math
from math import floor
from PIL import Image, ImageDraw
import random
import os
import torch
import numpy as np
import pickle
import yaml
from detectron2.config import get_cfg
from detectron2.structures import Instances, Boxes, pairwise_iou, BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import logging
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import copy

class VRDTrainData:
    """
    Register data for VRD training
    """
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.dataset_dicts = self._fetch_data_dict()
        self.register_dataset()
        try:
            statistics = self.get_statistics()
        except:
            pass

    def register_dataset(self, dataloader=False):
        """
        Register datasets to use with Detectron2
        """
        if not dataloader:
            DatasetCatalog.register('VRD_{}'.format(self.split), lambda: self.dataset_dicts)
        else:    
            DatasetCatalog.register('VRD_{}'.format(self.split), lambda: self.dataloader)            
            
        #Get labels
        self.mapping_dictionary = json.load(open(self.cfg.DATASETS.VRD.MAPPING_DICTIONARY, 'r'))
        self.idx_to_classes = sorted(self.mapping_dictionary['label_to_idx'], key=lambda k: self.mapping_dictionary['label_to_idx'][k])
        self.idx_to_predicates = sorted(self.mapping_dictionary['predicate_to_idx'], key=lambda k: self.mapping_dictionary['predicate_to_idx'][k])
        MetadataCatalog.get('VRD_{}'.format(self.split)).set(thing_classes=self.idx_to_classes, predicate_classes=self.idx_to_predicates)
    
    def _fetch_data_dict(self):
        """
        Load data in detectron format
        """
        fileName = "tmp/vrd_{}_data.pkl".format(self.split)
        print("Loading file: ", fileName)
        if os.path.isfile(fileName):
            #If data has been processed earlier, load that to save time
            with open(fileName, 'rb') as inputFile:
                dataset_dicts = pickle.load(inputFile)
        else:
            #Process data
            os.makedirs('tmp', exist_ok=True)
            dataset_dicts = self._process_data()
            with open(fileName, 'wb') as inputFile:
                pickle.dump(dataset_dicts, inputFile)
        return dataset_dicts
            
    def _process_data(self):
        if self.split == 'train':
            h5_file = h5py.File(self.cfg.DATASETS.VRD.VRD_H5, 'r')
            self.VRD_h5 = h5_file['train_group']
            file = json.load(open(os.path.join(self.cfg.DATASETS.VRD.IMAGE_DATA, 'detections_train.json'), 'r'))
            self.image_data = file['images']
        elif self.split == 'test':
            h5_file = h5py.File(self.cfg.DATASETS.VRD.VRD_H5, 'r')
            self.VRD_h5 = h5_file['test_group']
            file = json.load(open(os.path.join(self.cfg.DATASETS.VRD.IMAGE_DATA, 'detections_val.json'), 'r'))
            self.image_data = file['images']
        dataset_dicts = self._load_graphs()
        return dataset_dicts

    def get_statistics(self, eps=1e-3, bbox_overlap=True):
        num_object_classes = len(MetadataCatalog.get('VRD_{}'.format(self.split)).thing_classes) + 1
        num_relation_classes = len(MetadataCatalog.get('VRD_{}'.format(self.split)).predicate_classes) + 1
        
        fg_matrix = np.zeros((num_object_classes, num_object_classes, num_relation_classes), dtype=np.int64)
        bg_matrix = np.zeros((num_object_classes, num_object_classes), dtype=np.int64)
        fg_rel_count = np.zeros((num_relation_classes), dtype=np.int64)
        for idx, data in enumerate(self.dataset_dicts):
            gt_relations = data['relations']
            gt_classes = np.array([x['category_id'] for x in data['annotations']])
            gt_boxes = np.array([x['bbox'] for x in data['annotations']])
            for (o1, o2), rel in zip(gt_classes[gt_relations[:,:2]], gt_relations[:,2]):
                fg_matrix[o1, o2, rel] += 1
                fg_rel_count[rel] += 1

            for (o1, o2) in gt_classes[np.array(box_filter(gt_boxes, must_overlap=bbox_overlap), dtype=int)]:
                bg_matrix[o1, o2] += 1
        bg_matrix += 1
        fg_matrix[:, :, -1] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'fg_rel_count': torch.from_numpy(fg_rel_count).float(),
            'obj_classes': self.idx_to_classes + ['__background__'],
            'rel_classes': self.idx_to_predicates + ['__background__'],
        }
        print (torch.from_numpy(fg_rel_count).float())
        MetadataCatalog.get('VRD_{}'.format(self.split)).set(statistics=result)
        return result

    def _load_graphs(self):
        """
        Parse examples and create dictionaries
        """
        if self.split == 'train':
            split_mask = np.ones(4000).astype(bool)
            image_path = os.path.join(self.cfg.DATASETS.VRD.IMAGES, 'train_images')
        elif self.split == 'test':
            split_mask = np.ones(1000).astype(bool)
            image_path = os.path.join(self.cfg.DATASETS.VRD.IMAGES, 'val_images')
            #Filter images without bounding boxes
        split_mask &= self.VRD_h5['img_to_first_box'][:] >= 0
        if self.cfg.DATASETS.VRD.FILTER_EMPTY_RELATIONS:
            split_mask &= self.VRD_h5['img_to_first_rel'][:] >= 0
        image_index = np.where(split_mask)[0]    
        
        split_mask = np.zeros_like(split_mask).astype(bool)
        split_mask[image_index] = True
        
        # Get box information
        all_labels = self.VRD_h5['labels'][:]
        all_boxes = self.VRD_h5['boxes_1024'][:]  # x1,y1,x2,y2
        assert np.all(all_boxes[:, :] >= 0)  # sanity check
        assert np.all(all_boxes[:, :2] <= all_boxes[:, 2:])  
        
        first_box_index = self.VRD_h5['img_to_first_box'][split_mask]
        last_box_index = self.VRD_h5['img_to_last_box'][split_mask]
        first_relation_index = self.VRD_h5['img_to_first_rel'][split_mask]
        last_relation_index = self.VRD_h5['img_to_last_rel'][split_mask]

        #Load relation labels
        all_relations = self.VRD_h5['relationships'][:]
        all_relation_predicates = self.VRD_h5['predicates'][:]
        
        image_indexer = np.arange(len(self.image_data))[split_mask]
        # Iterate over images
        dataset_dicts = []
        num_rels = []
        num_objs = []
        for idx, _ in enumerate(image_index):
            record = {}
            #Get image metadata
            image_data = self.image_data[image_indexer[idx]]
            record['file_name'] = os.path.join(image_path, image_data['file_name'])
            record['image_id'] = image_data['id']
            record['height'] = image_data['height']
            record['width'] = image_data['width']
    
            #Get annotations
            boxes = all_boxes[first_box_index[idx]:last_box_index[idx] + 1, :]
            gt_classes = all_labels[first_box_index[idx]:last_box_index[idx] + 1]

            if first_relation_index[idx] > -1:
                predicates = all_relation_predicates[first_relation_index[idx]:last_relation_index[idx] + 1]
                objects = all_relations[first_relation_index[idx]:last_relation_index[idx] + 1] - first_box_index[idx] # 得到的是相对box编号
                predicates = predicates # 从0开始，最后一个类别是non-relation
                relations = np.column_stack((objects, predicates))
            else:
                assert not self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS
                relations = np.zeros((0, 3), dtype=np.int32)
            
            if self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP and self.split == 'train':
                # Remove boxes that don't overlap
                boxes_list = Boxes(boxes)
                ious = pairwise_iou(boxes_list, boxes_list)
                relation_boxes_ious = ious[relations[:,0], relations[:,1]]
                iou_indexes = np.where(relation_boxes_ious > 0.0)[0]
                if iou_indexes.size > 0:
                    relations = relations[iou_indexes]
                else:
                    #Ignore image
                    continue
            #Get masks if possible
            record['relations'] = relations
            objects = []
            for obj_idx in range(len(boxes)):
                resized_box = boxes[obj_idx] / self.cfg.DATASETS.VRD.BOX_SCALE * max(record['height'], record['width'])
                obj = {
                      "bbox": resized_box.tolist(),
                      "bbox_mode": BoxMode.XYXY_ABS,
                      "category_id": gt_classes[obj_idx],        
                }
                objects.append(obj)
            num_objs.append(len(objects))
            num_rels.append(len(relations))  
            record['annotations'] = objects
            dataset_dicts.append(record)
        print ("Max Rels:", np.max(num_rels), "Max Objs:", np.max(num_objs))
        print ("Avg Rels:", np.mean(num_rels), "Avg Objs:", np.mean(num_objs))
        print ("Median Rels:", np.median(num_rels), "Median Objs:", np.median(num_objs))
        return dataset_dicts

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(float), boxes.astype(float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    return inter